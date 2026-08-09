"""
Partial Map Export (`src/sonic/export_partial_map_data.py`)

Builds `web/sonic_map/map_data.json` from videos that already have ASR JSON
outputs, so UI work can continue while long-running ASR jobs are still in progress.
"""

import argparse
import json
import time
import hashlib
from pathlib import Path
from typing import Dict, List, Optional

from src.sonic.ASR import _parse_whisper_json, align_asr_with_segments
from src.sonic.annotations import (
    align_narrations_with_segments,
    get_narrations_debug_info,
    load_narrations_for_video,
)
from src.sonic.dataset import get_fake_shop_videos, get_video_file_path
from src.sonic.features import process_video_audio_segments
from src.sonic.reduction import reduce_mfccs_to_2d

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ASR_DIR = Path("/app/2026/audio_cache/asr")
DEFAULT_OUTPUT = WORKSPACE_ROOT / "web" / "sonic_map" / "map_data.json"
DEFAULT_CACHE_DIR = Path("/app/2026/audio_cache/partial_map_cache")


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _log(msg: str) -> None:
    print(f"[{_now()}] {msg}")


def _safe_parse_asr_json(json_path: Path) -> Optional[List[Dict]]:
    try:
        return _parse_whisper_json(json_path)
    except Exception as exc:
        _log(f"WARN parse failed for {json_path.name}: {exc}")
        return None


def _fingerprint_for_video(
    uid: str,
    video_path: Path,
    asr_json: Path,
    window_sec: float,
    hop_sec: float,
    narr_info: Dict[str, object],
) -> str:
    v_st = video_path.stat()
    a_st = asr_json.stat()
    narr_sig = "none"
    if narr_info.get("exists"):
        narr_sig = f"{narr_info.get('path')}|{narr_info.get('size')}|{narr_info.get('mtime')}"

    raw = "|".join(
        [
            uid,
            str(window_sec),
            str(hop_sec),
            str(video_path),
            str(v_st.st_size),
            str(v_st.st_mtime),
            str(asr_json),
            str(a_st.st_size),
            str(a_st.st_mtime),
            narr_sig,
        ]
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _cache_file_for_uid(cache_dir: Path, uid: str) -> Path:
    return cache_dir / f"{uid}.json"


def _load_cached_segments(cache_path: Path, expected_fp: str) -> Optional[Dict]:
    if not cache_path.exists() or cache_path.stat().st_size == 0:
        return None
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return None

    if payload.get("fingerprint") != expected_fp:
        return None
    if not isinstance(payload.get("segments"), list):
        return None
    return payload


def _write_cached_segments(cache_path: Path, payload: Dict) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    tmp.replace(cache_path)


def _collect_ready_video_entries(asr_dir: Path, limit: Optional[int]) -> List[Dict]:
    all_videos = get_fake_shop_videos(limit=None)
    ready = []
    scanned = 0

    _log(f"Scanning metadata for ASR-ready videos in {asr_dir} ...")
    for meta in all_videos:
        scanned += 1
        uid = meta.get("video_uid")
        if not uid:
            continue

        asr_json = asr_dir / f"{uid}.json"
        v_path = get_video_file_path(uid)
        if not v_path:
            continue
        if not asr_json.exists() or asr_json.stat().st_size == 0:
            continue

        ready.append({
            "meta": meta,
            "uid": uid,
            "video_path": v_path,
            "asr_json": asr_json,
        })

        if limit and len(ready) >= limit:
            break

        if scanned % 100 == 0:
            _log(f"Scanned {scanned} metadata entries; ready={len(ready)}")

    _log(f"Ready videos found: {len(ready)} (scanned {scanned})")
    return ready


def build_partial_map(
    limit: Optional[int],
    window_sec: float,
    hop_sec: float,
    asr_dir: Path,
    output_path: Path,
    reduce_method: str,
    cache_dir: Path,
    cache_read: bool,
    cache_write: bool,
) -> Dict:
    start = time.time()
    ready = _collect_ready_video_entries(asr_dir=asr_dir, limit=limit)
    if not ready:
        raise RuntimeError("No ASR-ready videos found yet. Nothing to export.")

    narr_info = get_narrations_debug_info()
    _log(f"Narrations source: {narr_info}")

    all_segments: List[Dict] = []
    processed_videos: List[Dict] = []
    total = len(ready)
    cache_hits = 0
    cache_misses = 0

    for idx, item in enumerate(ready, start=1):
        uid = item["uid"]
        meta = item["meta"]
        v_path = item["video_path"]
        asr_json = item["asr_json"]
        v_start = time.time()

        _log(f"[{idx}/{total}] Processing {uid}")
        _log(f"[{idx}/{total}] video={v_path}")
        _log(f"[{idx}/{total}] asr_json={asr_json}")

        fp = _fingerprint_for_video(
            uid=uid,
            video_path=v_path,
            asr_json=asr_json,
            window_sec=window_sec,
            hop_sec=hop_sec,
            narr_info=narr_info,
        )
        cache_path = _cache_file_for_uid(cache_dir, uid)
        cached = _load_cached_segments(cache_path, expected_fp=fp) if cache_read else None
        if cached is not None:
            cache_hits += 1
            segments = cached["segments"]
            processed_videos.append(cached["video_summary"])
            all_segments.extend(segments)
            _log(
                f"[{idx}/{total}] CACHE HIT {uid}: segments={len(segments)} narrations={cached.get('narration_count', 'n/a')}"
            )
            elapsed = time.time() - v_start
            avg = (time.time() - start) / idx
            remain = max(total - idx, 0)
            eta = avg * remain
            _log(
                f"[{idx}/{total}] Done {uid} in {elapsed:.1f}s | cumulative_segments={len(all_segments)} | ETA~{eta/60.0:.1f}m"
            )
            continue

        cache_misses += 1
        _log(f"[{idx}/{total}] CACHE MISS {uid}")

        asr_transcripts = _safe_parse_asr_json(asr_json)
        if asr_transcripts is None:
            _log(f"[{idx}/{total}] SKIP {uid}: ASR JSON parse failed")
            continue

        _log(f"[{idx}/{total}] Parsed ASR phrases: {len(asr_transcripts)}")

        segments = process_video_audio_segments(v_path, window_sec=window_sec, hop_sec=hop_sec)
        _log(f"[{idx}/{total}] Audio segments extracted: {len(segments)}")

        segments = align_asr_with_segments(segments, asr_transcripts)
        _log(f"[{idx}/{total}] ASR aligned")

        narrations = load_narrations_for_video(uid)
        segments = align_narrations_with_segments(segments, narrations)
        narr_seg_count = sum(1 for s in segments if s.get("narration_text"))
        _log(
            f"[{idx}/{total}] Narrations loaded: {len(narrations)} | segments_with_narration={narr_seg_count}/{len(segments)}"
        )

        all_segments.extend(segments)
        video_summary = {
            "video_uid": uid,
            "duration_sec": meta.get("duration_sec", 0),
            "participant_id": meta.get("fb_participant_id"),
            "segment_count": len(segments),
            "relative_video_path": f"2026/v2/video_540ss/{uid}.mp4",
            "relative_audio_path": f"2026/audio_cache/{uid}.wav",
        }
        processed_videos.append(video_summary)

        if cache_write:
            _write_cached_segments(
                cache_path,
                {
                    "fingerprint": fp,
                    "video_uid": uid,
                    "asr_phrase_count": len(asr_transcripts),
                    "narration_count": len(narrations),
                    "segments": segments,
                    "video_summary": video_summary,
                },
            )
            _log(f"[{idx}/{total}] Wrote cache: {cache_path}")

        elapsed = time.time() - v_start
        avg = (time.time() - start) / idx
        remain = max(total - idx, 0)
        eta = avg * remain
        _log(
            f"[{idx}/{total}] Done {uid} in {elapsed:.1f}s | cumulative_segments={len(all_segments)} | ETA~{eta/60.0:.1f}m"
        )

    if not all_segments:
        raise RuntimeError("No segments generated from ready videos.")

    _log(
        f"Running dimensionality reduction with method={reduce_method} on {len(all_segments)} segments ..."
    )
    red_start = time.time()
    all_segments = reduce_mfccs_to_2d(all_segments, method=reduce_method)
    _log(f"Dimensionality reduction complete in {time.time() - red_start:.1f}s")
    _log(f"Cache stats: hits={cache_hits} misses={cache_misses} cache_dir={cache_dir}")

    payload = {
        "metadata": {
            "title": "Ego4D Fake Shop Acoustic Map (partial preview)",
            "scenario": "Grocery shopping indoors",
            "video_source": "frl_track_1_public",
            "total_videos": len(processed_videos),
            "total_segments": len(all_segments),
            "window_sec": window_sec,
            "hop_sec": hop_sec,
            "partial_preview": True,
            "generated_at": _now(),
            "reduction_method": reduce_method,
            "asr_source_dir": str(asr_dir),
            "cache_dir": str(cache_dir),
            "cache_hits": cache_hits,
            "cache_misses": cache_misses,
        },
        "videos": processed_videos,
        "segments": all_segments,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    total_elapsed = time.time() - start
    _log(
        f"Wrote partial map to {output_path} | videos={len(processed_videos)} segments={len(all_segments)} size={output_path.stat().st_size/1024/1024:.2f}MB elapsed={total_elapsed/60.0:.1f}m"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export partial map_data.json from videos that already have ASR JSON outputs"
    )
    parser.add_argument("--limit", type=int, default=0, help="Max ready videos to include (0=all ready)")
    parser.add_argument("--window", type=float, default=3.0, help="Audio sliding window duration")
    parser.add_argument("--hop", type=float, default=2.0, help="Audio sliding window hop")
    parser.add_argument(
        "--reduce-method",
        type=str,
        default="pca",
        choices=["pca", "tsne", "umap"],
        help="2D reduction method (pca is fastest for preview)",
    )
    parser.add_argument(
        "--asr-dir",
        type=Path,
        default=DEFAULT_ASR_DIR,
        help="Directory containing whisper JSON outputs",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output map_data.json path",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help="Per-video segment cache directory",
    )
    parser.add_argument(
        "--no-cache-read",
        action="store_true",
        help="Disable reading per-video cache",
    )
    parser.add_argument(
        "--no-cache-write",
        action="store_true",
        help="Disable writing per-video cache",
    )

    args = parser.parse_args()
    limit = None if args.limit <= 0 else args.limit

    _log("Starting partial map export...")
    _log(
        f"Config: limit={limit or 'all-ready'} window={args.window} hop={args.hop} reduce={args.reduce_method}"
    )
    _log(f"ASR dir: {args.asr_dir}")
    _log(f"Output: {args.output}")
    _log(f"Cache dir: {args.cache_dir} (read={not args.no_cache_read}, write={not args.no_cache_write})")

    build_partial_map(
        limit=limit,
        window_sec=args.window,
        hop_sec=args.hop,
        asr_dir=args.asr_dir,
        output_path=args.output,
        reduce_method=args.reduce_method,
        cache_dir=args.cache_dir,
        cache_read=not args.no_cache_read,
        cache_write=not args.no_cache_write,
    )


if __name__ == "__main__":
    main()
