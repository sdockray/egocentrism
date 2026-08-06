"""
Whisper.cpp ASR Integration Module (`src/sonic/ASR.py`)

Invokes local C++ whisper.cpp binary to perform fast, Metal-accelerated
speech transcription on WAV audio files and extracts timestamped transcripts.
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

WHISPER_DIR = Path.home() / "Documents" / "dev" / "prj" / "word-processor" / "whisper.cpp"
WHISPER_BIN = WHISPER_DIR / "main"
WHISPER_MODEL = WHISPER_DIR / "models" / "ggml-medium.en.bin"


def run_whisper_asr(
    wav_path: Path,
    threads: int = 4,
) -> List[Dict]:
    """
    Runs whisper.cpp on a WAV audio file and returns a list of timestamped transcript segments:
    [
        {
            "start_sec": 10.5,
            "end_sec": 14.2,
            "text": "How much for this item?"
        }, ...
    ]
    """
    if not WHISPER_BIN.exists():
        raise FileNotFoundError(f"whisper.cpp binary not found at {WHISPER_BIN}")
    if not WHISPER_MODEL.exists():
        raise FileNotFoundError(f"Whisper model not found at {WHISPER_MODEL}")
    if not wav_path.exists():
        raise FileNotFoundError(f"WAV audio file not found at {wav_path}")

    output_dir = wav_path.parent / "asr"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_prefix = output_dir / wav_path.stem
    json_path = Path(f"{out_prefix}.json")

    if json_path.exists() and json_path.stat().st_size > 0:
        return _parse_whisper_json(json_path)

    cmd = [
        str(WHISPER_BIN),
        "-m",
        str(WHISPER_MODEL),
        "-f",
        str(wav_path),
        "-t",
        str(threads),
        "-oj",
        "-of",
        str(out_prefix),
        "-np",
    ]

    print(f"Running ASR on {wav_path.name} via whisper.cpp...")
    subprocess.run(cmd, check=True)

    if not json_path.exists():
        raise RuntimeError(f"Whisper JSON output expected at {json_path} but not found.")

    return _parse_whisper_json(json_path)


def _parse_whisper_json(json_path: Path) -> List[Dict]:
    """Parses whisper.json output into clean segment list."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    raw_segments = data.get("transcription", [])
    parsed = []

    for seg in raw_segments:
        text = seg.get("text", "").strip()
        if not text:
            continue

        offsets = seg.get("timestamps", {})
        start_sec = None
        end_sec = None

        if "from" in offsets and "to" in offsets:
            from_val = offsets["from"]
            to_val = offsets["to"]
            start_sec = _parse_timestamp_str(from_val)
            end_sec = _parse_timestamp_str(to_val)
        else:
            start_sec = seg.get("offsets", {}).get("from", 0) / 1000.0
            end_sec = seg.get("offsets", {}).get("to", 0) / 1000.0

        parsed.append(
            {
                "start_sec": round(start_sec, 2),
                "end_sec": round(end_sec, 2),
                "text": text,
            }
        )

    return parsed


def _parse_timestamp_str(ts) -> float:
    """Helper to convert HH:MM:SS,mmm or ms int/float into seconds float."""
    if isinstance(ts, (int, float)):
        return ts / 1000.0 if ts > 10000 else float(ts)
    
    if isinstance(ts, str):
        clean_ts = ts.replace(",", ".")
        parts = clean_ts.split(":")
        if len(parts) == 3:
            h, m, s = parts
            return float(h) * 3600.0 + float(m) * 60.0 + float(s)
        elif len(parts) == 2:
            m, s = parts
            return float(m) * 60.0 + float(s)
        else:
            return float(clean_ts)
    return 0.0


def align_asr_with_segments(
    segments: List[Dict],
    asr_transcripts: List[Dict],
    min_overlap_sec: float = 0.8,
) -> List[Dict]:
    """
    Overlays ASR speech transcripts onto sliding audio segments with strict overlap criteria.
    Requires at least `min_overlap_sec` overlap between speech phrase and segment interval.
    """
    for seg in segments:
        seg_start = seg["start_sec"]
        seg_end = seg["end_sec"]

        overlapping_texts = []
        seen = set()

        for asr in asr_transcripts:
            a_start = asr["start_sec"]
            a_end = asr["end_sec"]

            overlap_start = max(seg_start, a_start)
            overlap_end = min(seg_end, a_end)
            overlap_dur = overlap_end - overlap_start

            if overlap_dur >= min_overlap_sec:
                txt = asr["text"].strip()
                clean_key = txt.lower()
                if clean_key not in seen:
                    seen.add(clean_key)
                    overlapping_texts.append(txt)

        seg["asr_text"] = " ".join(overlapping_texts) if overlapping_texts else ""

    return segments


if __name__ == "__main__":
    from src.sonic.dataset import get_fake_shop_videos, get_video_file_path
    from src.sonic.features import extract_audio_wav

    v_list = get_fake_shop_videos(limit=1)
    target_uid = v_list[0]["video_uid"]
    v_path = get_video_file_path(target_uid)

    if v_path:
        wav_p = extract_audio_wav(v_path)
        print(f"Testing ASR on {wav_p}...")
        asr_res = run_whisper_asr(wav_p)
        print(f"Extracted {len(asr_res)} ASR speech segments.")
        print("Sample transcript:", json.dumps(asr_res[:3], indent=2))
