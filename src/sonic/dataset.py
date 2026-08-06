"""
Fake Shop Dataset & Download Utilities (`src/sonic/dataset.py`)

Identifies, filters, and manages video files for egocentric videos recorded
in the Facebook Reality Labs (FRL) Fake Shop environment ('Grocery shopping indoors').
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Root directory of workspace
WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
EGO4D_JSON_PATH = WORKSPACE_ROOT / "2026" / "ego4d.json"
EGO4D_CATALOG_PATH = WORKSPACE_ROOT / "2026" / "fake_shop_videos.json"
VIDEO_DIR = Path(os.getenv("SCRATCH_DIR", "/mnt/data-volume")) / "v2" / "video_540ss"
TARGET_ANCHOR_UID = "0049fdd8-0044-4ef5-9c34-b3469416ebe5"


def _build_fake_shop_catalog(all_videos: List[Dict]) -> List[Dict]:
    """Extract the small subset of metadata this Sonic pipeline actually needs."""
    catalog = []

    for video in all_videos:
        if video.get("video_source") != "frl_track_1_public":
            continue

        scenarios = video.get("scenarios") or []
        if "Grocery shopping indoors" not in scenarios:
            continue

        catalog.append(
            {
                "video_uid": video.get("video_uid"),
                "video_source": video.get("video_source"),
                "scenarios": scenarios,
                "duration_sec": video.get("duration_sec", 0),
                "fb_participant_id": video.get("fb_participant_id"),
            }
        )

    return catalog


def _write_fake_shop_catalog(videos: List[Dict]) -> None:
    EGO4D_CATALOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(EGO4D_CATALOG_PATH, "w", encoding="utf-8") as f:
        json.dump({"videos": videos}, f)


def _read_videos_payload(path: Path) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        return data.get("videos", [])
    if isinstance(data, list):
        return data
    raise ValueError(f"Unexpected metadata format in {path}")


def load_ego4d_metadata() -> List[Dict]:
    """Loads a compact fake-shop metadata catalog, deriving it if needed."""
    if EGO4D_CATALOG_PATH.exists():
        return _read_videos_payload(EGO4D_CATALOG_PATH)

    if not EGO4D_JSON_PATH.exists():
        raise FileNotFoundError(
            "No Sonic metadata catalog found. Expected either "
            f"{EGO4D_CATALOG_PATH} or the full Ego4D metadata at {EGO4D_JSON_PATH}"
        )

    all_videos = _read_videos_payload(EGO4D_JSON_PATH)
    catalog = _build_fake_shop_catalog(all_videos)
    _write_fake_shop_catalog(catalog)
    return catalog


def get_fake_shop_videos(limit: Optional[int] = None) -> List[Dict]:
    """
    Returns metadata entries for videos recorded in the FRL Fake Shop
    ('Grocery shopping indoors' scenario under 'frl_track_1_public').
    Always prioritizes TARGET_ANCHOR_UID as the first element.
    If limit is None or <= 0, returns all matching fake shop videos.
    """
    all_videos = load_ego4d_metadata()
    fake_shop_videos = []

    anchor_video = None

    for v in all_videos:
        uid = v.get("video_uid")

        if uid == TARGET_ANCHOR_UID:
            anchor_video = v
        else:
            fake_shop_videos.append(v)

    selected = []
    if anchor_video:
        selected.append(anchor_video)
    
    if limit is not None and limit > 0:
        selected.extend(fake_shop_videos[: max(0, limit - len(selected))])
    else:
        selected.extend(fake_shop_videos)

    return selected


def get_video_file_path(video_uid: str) -> Optional[Path]:
    """Returns local Path to video MP4 if present, else None."""
    expected_path = VIDEO_DIR / f"{video_uid}.mp4"
    if expected_path.exists() and expected_path.stat().st_size > 0:
        return expected_path
    return None


def download_missing_fake_shop_videos(video_list: List[Dict]) -> List[Path]:
    """
    Checks local existence of videos in video_list.
    Downloads missing videos via the official ego4d CLI.
    """
    import subprocess

    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    local_paths = []
    missing_uids = []

    for v in video_list:
        uid = v["video_uid"]
        local_p = get_video_file_path(uid)
        if local_p:
            local_paths.append(local_p)
        else:
            missing_uids.append(uid)

    if missing_uids:
        print(f"Found {len(missing_uids)} missing video(s) to download.")
        uid_file = WORKSPACE_ROOT / "2026" / "missing_fake_shop_uids.txt"
        with open(uid_file, "w") as f:
            f.write("\n".join(missing_uids) + "\n")

        cmd = [
            "ego4d",
            "-y",
            "--aws_profile_name=ego4d",
            "--datasets",
            "video_540ss",
            "--video_uid_file",
            str(uid_file),
            "-o",
            str(WORKSPACE_ROOT / "2026"),
        ]
        print(f"Executing ego4d download: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

        for uid in missing_uids:
            local_p = get_video_file_path(uid)
            if local_p:
                local_paths.append(local_p)

    else:
        print("All target videos are already available locally.")

    return local_paths


if __name__ == "__main__":
    videos = get_fake_shop_videos(limit=15)
    print(f"Cataloged {len(videos)} fake shop videos.")
    for idx, v in enumerate(videos):
        uid = v["video_uid"]
        dur = v.get("duration_sec", 0) / 60.0
        local_p = get_video_file_path(uid)
        status = f"LOCAL ({local_p})" if local_p else "MISSING"
        print(f" [{idx+1:02d}] {uid} | {dur:4.1f} min | {status}")
