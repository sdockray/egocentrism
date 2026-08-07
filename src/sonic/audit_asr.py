"""
ASR Audit Utility (`src/sonic/audit_asr.py`)

Scans Whisper JSON outputs and flags likely transcription-loop failures,
such as repeated phrases over long durations. Produces:
1) A JSON report with per-video diagnostics.
2) A UID list file for targeted re-transcription.
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ASR_DIR = WORKSPACE_ROOT / "2026" / "audio_cache" / "asr"
DEFAULT_REPORT_PATH = DEFAULT_ASR_DIR / "asr_audit_report.json"
DEFAULT_RETRANSCRIBE_UID_PATH = WORKSPACE_ROOT / "2026" / "asr_retranscribe_uids.txt"


def _parse_timestamp_str(ts) -> float:
    if isinstance(ts, (int, float)):
        return ts / 1000.0 if ts > 10000 else float(ts)

    if isinstance(ts, str):
        clean_ts = ts.replace(",", ".")
        parts = clean_ts.split(":")
        if len(parts) == 3:
            h, m, s = parts
            return float(h) * 3600.0 + float(m) * 60.0 + float(s)
        if len(parts) == 2:
            m, s = parts
            return float(m) * 60.0 + float(s)
        return float(clean_ts)

    return 0.0


def _normalize_text(text: str) -> str:
    lowered = text.lower().strip()
    cleaned = re.sub(r"[^a-z0-9\s]", " ", lowered)
    return re.sub(r"\s+", " ", cleaned).strip()


def _load_whisper_segments(json_path: Path) -> List[Dict]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    raw_segments = []
    if isinstance(data, dict):
        raw_segments = data.get("transcription", [])
    elif isinstance(data, list):
        raw_segments = data

    parsed = []
    for seg in raw_segments:
        if not isinstance(seg, dict):
            continue

        text = str(seg.get("text", "")).strip()
        if not text:
            continue

        offsets = seg.get("timestamps", {})
        if isinstance(offsets, dict) and "from" in offsets and "to" in offsets:
            start_sec = _parse_timestamp_str(offsets["from"])
            end_sec = _parse_timestamp_str(offsets["to"])
        else:
            start_sec = seg.get("offsets", {}).get("from", 0) / 1000.0
            end_sec = seg.get("offsets", {}).get("to", 0) / 1000.0

        parsed.append(
            {
                "start_sec": float(start_sec),
                "end_sec": float(end_sec),
                "text": text,
                "norm_text": _normalize_text(text),
            }
        )

    return parsed


def _max_consecutive_repeat(segments: List[Dict]) -> Tuple[int, str]:
    max_streak = 0
    max_phrase = ""
    curr_streak = 0
    curr_phrase = ""

    for seg in segments:
        phrase = seg["norm_text"]
        if phrase and phrase == curr_phrase:
            curr_streak += 1
        else:
            curr_phrase = phrase
            curr_streak = 1 if phrase else 0

        if curr_streak > max_streak:
            max_streak = curr_streak
            max_phrase = curr_phrase

    return max_streak, max_phrase


def _tail_repeat_stats(segments: List[Dict], tail_window_sec: float) -> Tuple[int, str, float, float]:
    if not segments:
        return 0, "", 0.0, 0.0

    end_time = segments[-1]["end_sec"]
    start_cutoff = max(0.0, end_time - tail_window_sec)
    tail_segments = [s for s in segments if s["start_sec"] >= start_cutoff and s["norm_text"]]

    if not tail_segments:
        return 0, "", 0.0, 0.0

    counts = Counter(s["norm_text"] for s in tail_segments)
    top_phrase, top_count = counts.most_common(1)[0]
    ratio = top_count / len(tail_segments)
    tail_duration = tail_segments[-1]["end_sec"] - tail_segments[0]["start_sec"]
    return top_count, top_phrase, ratio, max(0.0, tail_duration)


def audit_transcript(
    segments: List[Dict],
    min_segments: int,
    min_consecutive_repeat: int,
    tail_window_sec: float,
    tail_repeat_ratio: float,
    min_tail_repeat_count: int,
    min_tail_duration_sec: float,
) -> Dict:
    if not segments:
        return {
            "flagged": True,
            "reasons": ["empty_transcript"],
            "segment_count": 0,
        }

    duration_sec = max(0.0, segments[-1]["end_sec"] - segments[0]["start_sec"])
    phrase_counts = Counter(s["norm_text"] for s in segments if s["norm_text"])
    unique_phrases = len(phrase_counts)
    top_phrase = ""
    top_count = 0
    top_ratio = 0.0
    if phrase_counts:
        top_phrase, top_count = phrase_counts.most_common(1)[0]
        top_ratio = top_count / len(segments)

    max_streak, streak_phrase = _max_consecutive_repeat(segments)
    tail_count, tail_phrase, tail_ratio, tail_duration = _tail_repeat_stats(segments, tail_window_sec)

    reasons = []
    if len(segments) < min_segments:
        reasons.append("too_few_segments")

    if max_streak >= min_consecutive_repeat:
        reasons.append("long_consecutive_repeat")

    if (
        tail_count >= min_tail_repeat_count
        and tail_ratio >= tail_repeat_ratio
        and tail_duration >= min_tail_duration_sec
    ):
        reasons.append("tail_repeat_dominance")

    flagged = any(r in reasons for r in ("empty_transcript", "long_consecutive_repeat", "tail_repeat_dominance"))

    return {
        "flagged": flagged,
        "reasons": reasons,
        "segment_count": len(segments),
        "duration_sec": round(duration_sec, 2),
        "unique_phrase_count": unique_phrases,
        "top_phrase": top_phrase,
        "top_phrase_count": top_count,
        "top_phrase_ratio": round(top_ratio, 4),
        "max_consecutive_repeat": max_streak,
        "max_consecutive_phrase": streak_phrase,
        "tail_top_phrase": tail_phrase,
        "tail_top_count": tail_count,
        "tail_top_ratio": round(tail_ratio, 4),
        "tail_duration_sec": round(tail_duration, 2),
    }


def run_audit(
    asr_dir: Path,
    report_path: Path,
    uid_output_path: Path,
    min_segments: int,
    min_consecutive_repeat: int,
    tail_window_sec: float,
    tail_repeat_ratio: float,
    min_tail_repeat_count: int,
    min_tail_duration_sec: float,
) -> Dict:
    files = sorted(asr_dir.glob("*.json"))
    results = []
    flagged_uids = []

    for json_file in files:
        try:
            segments = _load_whisper_segments(json_file)
            analysis = audit_transcript(
                segments=segments,
                min_segments=min_segments,
                min_consecutive_repeat=min_consecutive_repeat,
                tail_window_sec=tail_window_sec,
                tail_repeat_ratio=tail_repeat_ratio,
                min_tail_repeat_count=min_tail_repeat_count,
                min_tail_duration_sec=min_tail_duration_sec,
            )
            uid = json_file.stem
            row = {
                "video_uid": uid,
                "asr_json_path": str(json_file),
                **analysis,
            }
            results.append(row)
            if analysis["flagged"]:
                flagged_uids.append(uid)
        except Exception as exc:  # keep audit robust even with malformed files
            results.append(
                {
                    "video_uid": json_file.stem,
                    "asr_json_path": str(json_file),
                    "flagged": True,
                    "reasons": [f"parse_error: {exc}"],
                }
            )
            flagged_uids.append(json_file.stem)

    report = {
        "summary": {
            "asr_dir": str(asr_dir),
            "total_files": len(files),
            "flagged_files": len(flagged_uids),
            "clean_files": max(0, len(files) - len(flagged_uids)),
        },
        "thresholds": {
            "min_segments": min_segments,
            "min_consecutive_repeat": min_consecutive_repeat,
            "tail_window_sec": tail_window_sec,
            "tail_repeat_ratio": tail_repeat_ratio,
            "min_tail_repeat_count": min_tail_repeat_count,
            "min_tail_duration_sec": min_tail_duration_sec,
        },
        "flagged_video_uids": sorted(set(flagged_uids)),
        "results": results,
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    uid_output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(uid_output_path, "w", encoding="utf-8") as f:
        for uid in sorted(set(flagged_uids)):
            f.write(uid + "\n")

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Whisper transcripts for repetition-loop failures")
    parser.add_argument("--asr-dir", type=Path, default=DEFAULT_ASR_DIR, help="Directory containing whisper JSON transcripts")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH, help="Path to write JSON audit report")
    parser.add_argument(
        "--retranscribe-uids",
        type=Path,
        default=DEFAULT_RETRANSCRIBE_UID_PATH,
        help="Path to write newline-delimited UIDs flagged for re-transcription",
    )
    parser.add_argument("--min-segments", type=int, default=20, help="Minimum transcript segments expected for a normal run")
    parser.add_argument(
        "--min-consecutive-repeat",
        type=int,
        default=25,
        help="Flag if one normalized phrase repeats this many times consecutively",
    )
    parser.add_argument("--tail-window-sec", type=float, default=900.0, help="Tail window to inspect for phrase repetition dominance")
    parser.add_argument(
        "--tail-repeat-ratio",
        type=float,
        default=0.65,
        help="Flag if a single phrase dominates this fraction of tail segments",
    )
    parser.add_argument(
        "--min-tail-repeat-count",
        type=int,
        default=40,
        help="Minimum count for top tail phrase before tail-repeat rule can trigger",
    )
    parser.add_argument(
        "--min-tail-duration-sec",
        type=float,
        default=300.0,
        help="Minimum tail coverage duration before tail-repeat rule can trigger",
    )

    args = parser.parse_args()

    if not args.asr_dir.exists():
        raise FileNotFoundError(f"ASR directory not found: {args.asr_dir}")

    report = run_audit(
        asr_dir=args.asr_dir,
        report_path=args.report,
        uid_output_path=args.retranscribe_uids,
        min_segments=args.min_segments,
        min_consecutive_repeat=args.min_consecutive_repeat,
        tail_window_sec=args.tail_window_sec,
        tail_repeat_ratio=args.tail_repeat_ratio,
        min_tail_repeat_count=args.min_tail_repeat_count,
        min_tail_duration_sec=args.min_tail_duration_sec,
    )

    summary = report["summary"]
    print(
        "ASR audit complete: "
        f"{summary['flagged_files']} flagged / {summary['total_files']} total. "
        f"Report: {args.report} | Retranscribe UID list: {args.retranscribe_uids}"
    )


if __name__ == "__main__":
    main()
