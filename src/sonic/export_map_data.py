"""
Master Export Pipeline (`src/sonic/export_map_data.py`)

Processes all downloaded fake shop videos:
1. Extracts audio WAV & sliding window audio features (MFCCs, RMS, Centroid, ZCR).
2. Runs whisper.cpp ASR for high-precision speech transcripts.
3. Loads & aligns Ego4D official narrations.
4. Performs 2D t-SNE / PCA dimensionality reduction on 20D MFCC space.
5. Exports web-ready dataset JSON to `web/sonic_map/map_data.json`.
"""

import json
import sys
from pathlib import Path
from typing import List, Dict

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
WEB_OUTPUT_DIR = WORKSPACE_ROOT / "web" / "sonic_map"
WEB_JSON_PATH = WEB_OUTPUT_DIR / "map_data.json"

from src.sonic.dataset import get_fake_shop_videos, get_video_file_path, download_missing_fake_shop_videos
from src.sonic.features import extract_audio_wav, process_video_audio_segments
from src.sonic.ASR import run_whisper_asr, align_asr_with_segments
from src.sonic.annotations import load_narrations_for_video, align_narrations_with_segments
from src.sonic.reduction import reduce_mfccs_to_2d


def build_fake_shop_map_dataset(limit: int = 4, window_sec: float = 2.0, hop_sec: float = 1.0) -> Dict:
    """
    Builds the unified dataset for the Giant MFCC Map.
    Auto-downloads any missing target videos using the ego4d CLI.
    """
    videos_meta = get_fake_shop_videos(limit=limit)
    print(f"Targeting {len(videos_meta)} fake shop video(s)...")
    download_missing_fake_shop_videos(videos_meta)

    all_segments: List[Dict] = []
    processed_videos: List[Dict] = []

    for v_info in videos_meta:
        uid = v_info["video_uid"]
        v_path = get_video_file_path(uid)

        if not v_path:
            print(f"Skipping {uid} (video file not downloaded locally)")
            continue

        print(f"\nProcessing fake shop video: {uid} ({v_info.get('duration_sec', 0)/60.0:.1f} mins)")

        # 1. Feature extraction
        wav_path = extract_audio_wav(v_path)
        segments = process_video_audio_segments(v_path, window_sec=window_sec, hop_sec=hop_sec)
        print(f" -> Extracted {len(segments)} audio segments (window={window_sec}s, hop={hop_sec}s)")

        # 2. ASR Speech Transcription
        try:
            asr_transcripts = run_whisper_asr(wav_path)
            segments = align_asr_with_segments(segments, asr_transcripts)
            print(f" -> Aligned {len(asr_transcripts)} ASR speech phrases")
        except Exception as e:
            print(f" -> Warning: ASR failed for {uid}: {e}", file=sys.stderr)
            for seg in segments:
                seg["asr_text"] = ""

        # 3. Official Ego4D Narrations
        narrations = load_narrations_for_video(uid)
        segments = align_narrations_with_segments(segments, narrations)
        print(f" -> Aligned {len(narrations)} official narrations")

        all_segments.extend(segments)
        processed_videos.append({
            "video_uid": uid,
            "duration_sec": v_info.get("duration_sec", 0),
            "participant_id": v_info.get("fb_participant_id"),
            "segment_count": len(segments),
            "relative_video_path": f"2026/v2/video_540ss/{uid}.mp4",
            "relative_audio_path": f"2026/audio_cache/{uid}.wav"
        })

    print(f"\nTotal audio segments compiled: {len(all_segments)}")
    print("Performing 2D t-SNE dimensionality reduction on MFCC vectors...")
    all_segments = reduce_mfccs_to_2d(all_segments, method="tsne")

    dataset_payload = {
        "metadata": {
            "title": "Ego4D Fake Shop Acoustic Map",
            "scenario": "Grocery shopping indoors",
            "video_source": "frl_track_1_public",
            "total_videos": len(processed_videos),
            "total_segments": len(all_segments),
            "window_sec": window_sec,
            "hop_sec": hop_sec,
        },
        "videos": processed_videos,
        "segments": all_segments,
    }

    # Ensure output dir exists
    WEB_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(WEB_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(dataset_payload, f, indent=2)

    print(f"Successfully exported map dataset to {WEB_JSON_PATH} ({WEB_JSON_PATH.stat().st_size / 1024 / 1024:.2f} MB)")
    return dataset_payload


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export Fake Shop Acoustic Map Dataset")
    parser.add_argument("--limit", type=int, default=10, help="Number of videos to process (default: 10, set 0 or -1 for all)")
    parser.add_argument("--window", type=float, default=3.0, help="Audio sliding window duration in seconds (default: 3.0)")
    parser.add_argument("--hop", type=float, default=2.0, help="Audio window hop duration in seconds (default: 2.0)")
    parser.add_argument("--all", action="store_true", help="Process all available fake shop videos in 2026/ego4d.json")

    args = parser.parse_args()
    target_limit = None if args.all or args.limit <= 0 else args.limit

    build_fake_shop_map_dataset(limit=target_limit, window_sec=args.window, hop_sec=args.hop)
