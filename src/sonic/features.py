"""
Audio Feature Extraction Module (`src/sonic/features.py`)

Extracts MFCCs, RMS Energy, Spectral Centroid, and Zero-Crossing Rate (ZCR)
from egocentric video audio streams for dimensional mapping and acoustic metrics.
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import librosa
import numpy as np


def extract_audio_wav(video_path: Path, sample_rate: int = 16000) -> Path:
    """
    Extracts mono WAV audio from an MP4 video file using ffmpeg.
    Saves temporary WAV in 2026/audio_cache/ directory.
    """
    cache_dir = video_path.parents[2] / "audio_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    wav_path = cache_dir / f"{video_path.stem}.wav"

    if wav_path.exists() and wav_path.stat().st_size > 0:
        return wav_path

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-acodec",
        "pcm_s16le",
        str(wav_path),
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return wav_path


def compute_segment_features(
    y: np.ndarray,
    sr: int,
    n_mfcc: int = 20,
) -> Dict[str, float]:
    """
    Computes summary acoustic metrics for a single audio segment array:
    - 20 MFCC means
    - RMS Energy (mean & max)
    - Spectral Centroid (mean)
    - Zero Crossing Rate (mean)
    - Spectral Rolloff (mean)
    """
    if len(y) < sr * 0.1:  # ignore extremely short segments (<100ms)
        return {}

    # MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_means = np.mean(mfccs, axis=1)

    # RMS Energy
    rms = librosa.feature.rms(y=y)[0]
    rms_mean = float(np.mean(rms))
    rms_max = float(np.max(rms))

    # Spectral Centroid
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    centroid_mean = float(np.mean(centroid))

    # Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y=y)[0]
    zcr_mean = float(np.mean(zcr))

    # Spectral Rolloff
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    rolloff_mean = float(np.mean(rolloff))

    feat_dict = {
        "rms_mean": rms_mean,
        "rms_max": rms_max,
        "spectral_centroid": centroid_mean,
        "zero_crossing_rate": zcr_mean,
        "spectral_rolloff": rolloff_mean,
    }

    # Add MFCC features
    for idx, val in enumerate(mfcc_means):
        feat_dict[f"mfcc_{idx+1}"] = float(val)

    return feat_dict


def process_video_audio_segments(
    video_path: Path,
    window_sec: float = 2.0,
    hop_sec: float = 1.0,
    sample_rate: int = 16000,
) -> List[Dict]:
    """
    Extracts audio, splits into sliding window segments, and computes feature vectors.

    Returns a list of segment dicts:
    [
        {
            "segment_id": "0049fdd8_seg_0012",
            "video_uid": "0049fdd8-...",
            "start_sec": 12.0,
            "end_sec": 14.0,
            "duration_sec": 2.0,
            "features": { ... },
            "mfcc_vector": [ ... ]
        }, ...
    ]
    """
    video_uid = video_path.stem
    wav_path = extract_audio_wav(video_path, sample_rate=sample_rate)

    y, sr = librosa.load(wav_path, sr=sample_rate)
    total_sec = len(y) / sr

    window_samples = int(window_sec * sr)
    hop_samples = int(hop_sec * sr)

    segments = []
    seg_idx = 0

    for start_sample in range(0, len(y) - window_samples + 1, hop_samples):
        end_sample = start_sample + window_samples
        y_seg = y[start_sample:end_sample]

        start_sec = start_sample / sr
        end_sec = end_sample / sr

        feats = compute_segment_features(y_seg, sr=sr)
        if not feats:
            continue

        mfcc_vec = [feats[f"mfcc_{i+1}"] for i in range(20)]

        seg_dict = {
            "segment_id": f"{video_uid[:8]}_seg_{seg_idx:04d}",
            "video_uid": video_uid,
            "start_sec": round(start_sec, 2),
            "end_sec": round(end_sec, 2),
            "duration_sec": round(end_sec - start_sec, 2),
            "rms_mean": feats["rms_mean"],
            "rms_max": feats["rms_max"],
            "spectral_centroid": feats["spectral_centroid"],
            "zero_crossing_rate": feats["zero_crossing_rate"],
            "spectral_rolloff": feats["spectral_rolloff"],
            "mfcc_vector": mfcc_vec,
        }
        segments.append(seg_dict)
        seg_idx += 1

    return segments


if __name__ == "__main__":
    from src.sonic.dataset import get_fake_shop_videos, get_video_file_path

    v_list = get_fake_shop_videos(limit=1)
    target_uid = v_list[0]["video_uid"]
    v_path = get_video_file_path(target_uid)

    if v_path:
        print(f"Extracting features from {v_path}...")
        segs = process_video_audio_segments(v_path, window_sec=2.0, hop_sec=1.0)
        print(f"Processed {len(segs)} audio segments.")
        print("Sample segment:", json.dumps(segs[0], indent=2))
    else:
        print(f"Target video {target_uid} not found locally.")
