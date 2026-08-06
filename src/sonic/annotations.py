"""
Ego4D Annotations Alignment Module (`src/sonic/annotations.py`)

Loads narrations, summaries, and action labels from Ego4D annotations
and overlays them onto timestamped audio segments with strict temporal boundaries.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
NARRATIONS_JSON_PATH = WORKSPACE_ROOT / "2026" / "v2" / "annotations" / "narrations.json"


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
    if not NARRATIONS_JSON_PATH.exists():
        return []

    with open(NARRATIONS_JSON_PATH, "r", encoding="utf-8") as f:
        narr_data = json.load(f)

    entry = narr_data.get(video_uid, {})
    narrations = []

    if isinstance(entry, dict):
        for pass_key in ["narration_pass_1", "narration_pass_2"]:
            pass_dict = entry.get(pass_key)
            if isinstance(pass_dict, dict):
                n_list = pass_dict.get("narrations", [])
                for n in n_list:
                    txt = n.get("narration_text", "").strip()
                    if txt:
                        narrations.append(
                            {
                                "timestamp_sec": round(n.get("timestamp_sec", 0), 2),
                                "narration_text": txt,
                            }
                        )
    elif isinstance(entry, list):
        for n_item in entry:
            if isinstance(n_item, dict):
                if "narration_pass_1" in n_item or "narration_pass_2" in n_item:
                    pass1 = n_item.get("narration_pass_1", {}).get("narrations", [])
                    pass2 = n_item.get("narration_pass_2", {}).get("narrations", [])
                    for n in pass1 + pass2:
                        txt = n.get("narration_text", "").strip()
                        if txt:
                            narrations.append(
                                {
                                    "timestamp_sec": round(n.get("timestamp_sec", 0), 2),
                                    "narration_text": txt,
                                }
                            )
                elif "narration_text" in n_item:
                    txt = n_item.get("narration_text", "").strip()
                    if txt:
                        narrations.append(
                            {
                                "timestamp_sec": round(n_item.get("timestamp_sec", 0), 2),
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
