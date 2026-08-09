"""
Ego4D Annotations Alignment Module (`src/sonic/annotations.py`)

Loads narrations, summaries, and action labels from Ego4D annotations
and overlays them onto timestamped audio segments with strict temporal boundaries.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
NARRATIONS_JSON_PATH = WORKSPACE_ROOT / "2026" / "v2" / "annotations" / "narrations.json"

_NARRATIONS_INDEX: Optional[Dict[str, object]] = None
_NARRATIONS_SOURCE_PATH: Optional[Path] = None
_NARRATIONS_SOURCE_STATS: Optional[Dict[str, object]] = None


def _candidate_narrations_paths() -> List[Path]:
    env_override = os.getenv("NARRATIONS_JSON_PATH", "").strip()
    candidates = []
    if env_override:
        candidates.append(Path(env_override))

    candidates.extend(
        [
            NARRATIONS_JSON_PATH,
            Path("/app/2026/v2/annotations/narrations.json"),
            WORKSPACE_ROOT / "2026" / "annotations" / "narrations.json",
        ]
    )
    return candidates


def _resolve_narrations_path() -> Optional[Path]:
    for p in _candidate_narrations_paths():
        if p.exists() and p.is_file() and p.stat().st_size > 0:
            return p
    return None


def _index_from_payload(payload: object) -> Dict[str, object]:
    # Common shape: {"<video_uid>": {...passes...}, ...}
    if isinstance(payload, dict):
        # Alternate shape: {"videos": [{"video_uid": ..., ...}, ...]}
        if isinstance(payload.get("videos"), list):
            out: Dict[str, object] = {}
            for row in payload["videos"]:
                if not isinstance(row, dict):
                    continue
                uid = row.get("video_uid")
                if uid:
                    out[str(uid)] = row
            return out

        # Heuristic: if keys look like UIDs, treat dict as index.
        out = {}
        for k, v in payload.items():
            if isinstance(k, str) and "-" in k:
                out[k] = v
        if out:
            return out

    # Alternate shape: [{"video_uid": ..., ...}, ...]
    if isinstance(payload, list):
        out = {}
        for row in payload:
            if not isinstance(row, dict):
                continue
            uid = row.get("video_uid")
            if uid:
                out[str(uid)] = row
        return out

    return {}


def _ensure_narrations_index() -> Dict[str, object]:
    global _NARRATIONS_INDEX, _NARRATIONS_SOURCE_PATH, _NARRATIONS_SOURCE_STATS
    if _NARRATIONS_INDEX is not None:
        return _NARRATIONS_INDEX

    src = _resolve_narrations_path()
    if src is None:
        _NARRATIONS_INDEX = {}
        _NARRATIONS_SOURCE_PATH = None
        _NARRATIONS_SOURCE_STATS = {"exists": False}
        return _NARRATIONS_INDEX

    with open(src, "r", encoding="utf-8") as f:
        payload = json.load(f)

    idx = _index_from_payload(payload)
    st = src.stat()
    _NARRATIONS_INDEX = idx
    _NARRATIONS_SOURCE_PATH = src
    _NARRATIONS_SOURCE_STATS = {
        "exists": True,
        "path": str(src),
        "size": st.st_size,
        "mtime": st.st_mtime,
        "indexed_videos": len(idx),
    }
    return _NARRATIONS_INDEX


def get_narrations_debug_info() -> Dict[str, object]:
    _ensure_narrations_index()
    return dict(_NARRATIONS_SOURCE_STATS or {"exists": False})


def _extract_narration_records(entry: object) -> List[Dict]:
    narrations: List[Dict] = []

    if isinstance(entry, dict):
        # Direct pass layout.
        for pass_key in ("narration_pass_1", "narration_pass_2"):
            pass_dict = entry.get(pass_key)
            if isinstance(pass_dict, dict):
                n_list = pass_dict.get("narrations", [])
                if isinstance(n_list, list):
                    for n in n_list:
                        if isinstance(n, dict):
                            narrations.append(n)

        # Already-flat layout.
        if not narrations and isinstance(entry.get("narrations"), list):
            for n in entry.get("narrations", []):
                if isinstance(n, dict):
                    narrations.append(n)

    elif isinstance(entry, list):
        for item in entry:
            if not isinstance(item, dict):
                continue
            if "narration_pass_1" in item or "narration_pass_2" in item:
                for pass_key in ("narration_pass_1", "narration_pass_2"):
                    pass_dict = item.get(pass_key, {})
                    if isinstance(pass_dict, dict):
                        n_list = pass_dict.get("narrations", [])
                        if isinstance(n_list, list):
                            for n in n_list:
                                if isinstance(n, dict):
                                    narrations.append(n)
            elif "narration_text" in item:
                narrations.append(item)

    return narrations


def load_narrations_for_video(video_uid: str) -> List[Dict]:
    """
    Loads narrations for a video UID from narrations.json.
    Returns sorted list of narration dicts:
    [
        {
            "timestamp_sec": 1017.0,
            "narration_text": "#C C walks around"
        }, ...
    ]
    """
    narr_data = _ensure_narrations_index()
    entry = narr_data.get(video_uid)
    if entry is None:
        return []

    narrations = []
    for n in _extract_narration_records(entry):
        txt = str(n.get("narration_text", "")).strip()
        if not txt:
            continue
        narrations.append(
            {
                "timestamp_sec": round(float(n.get("timestamp_sec", 0.0)), 2),
                "narration_text": txt,
            }
        )

    narrations.sort(key=lambda x: x["timestamp_sec"])
    return narrations


def align_narrations_with_segments(
    segments: List[Dict],
    narrations: List[Dict],
    margin_sec: float = 0.25,
) -> List[Dict]:
    """
    Strictly aligns narrations that occur within [start_sec - margin, end_sec + margin]
    of each segment. Deduplicates redundant narration strings.
    """
    for seg in segments:
        s_start = seg["start_sec"] - margin_sec
        s_end = seg["end_sec"] + margin_sec

        matches = []
        seen = set()
        for n in narrations:
            n_time = n["timestamp_sec"]
            if s_start <= n_time <= s_end:
                txt = n["narration_text"]
                clean_key = txt.lower()
                if clean_key not in seen:
                    seen.add(clean_key)
                    matches.append(txt)

        seg["narration_text"] = " | ".join(matches) if matches else ""
        seg["action_category"] = categorize_narration_action(seg["narration_text"], seg.get("asr_text", ""))

    return segments


def categorize_narration_action(narration_text: str, asr_text: str) -> str:
    """Categorizes action/sound into semantic group for color-coding in the map."""
    combined = (narration_text + " " + asr_text).lower()

    if any(k in combined for k in ["talk", "speak", "voice", "person x", "person y", "cashier", "say", "hello", "hi", "thank"]):
        return "Speech / Interaction"
    elif any(k in combined for k in ["basket", "cart", "item", "shelf", "pick", "grab", "touch", "put", "take", "hold"]):
        return "Object Interaction"
    elif any(k in combined for k in ["walk", "turn", "look around", "step", "move"]):
        return "Locomotion / Movement"
    elif any(k in combined for k in ["money", "cash", "receipt", "counter", "pay", "card"]):
        return "Checkout / Transaction"
    elif not combined.strip():
        return "Ambient Noise / Gap"
    else:
        return "Other Action"


if __name__ == "__main__":
    n_list = load_narrations_for_video("0049fdd8-0044-4ef5-9c34-b3469416ebe5")
    print(f"Loaded {len(n_list)} narrations for target video.")
    if n_list:
        print("Sample narration:", n_list[0])
