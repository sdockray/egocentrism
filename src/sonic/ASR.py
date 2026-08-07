"""
Whisper.cpp ASR Integration Module (`src/sonic/ASR.py`)

Invokes local C++ whisper.cpp binary to perform fast, Metal-accelerated
speech transcription on WAV audio files and extracts timestamped transcripts.
"""

import json
import os
import subprocess
import wave
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

DEFAULT_WHISPER_DIR = Path(os.getenv("WHISPER_DIR", "/opt/whisper.cpp"))
DEFAULT_WHISPER_MODEL = Path(
    os.getenv("WHISPER_MODEL", "/data/scratch/whisper-models/ggml-medium.en.bin")
)
DEFAULT_WHISPER_RETRY_MODEL = os.getenv("WHISPER_RETRY_MODEL", "base.en")
DEFAULT_WHISPER_MAX_RETRIES = int(os.getenv("WHISPER_MAX_RETRIES", "1"))

DEFAULT_WHISPER_TEMPERATURE = float(os.getenv("WHISPER_TEMPERATURE", "0"))
DEFAULT_WHISPER_TEMPERATURE_INC = float(os.getenv("WHISPER_TEMPERATURE_INC", "0.2"))
DEFAULT_WHISPER_RETRY_TEMPERATURE_BUMP = float(
    os.getenv("WHISPER_RETRY_TEMPERATURE_BUMP", "0.4")
)
DEFAULT_WHISPER_ENTROPY_THOLD = float(os.getenv("WHISPER_ENTROPY_THOLD", "2.2"))
DEFAULT_WHISPER_LOGPROB_THOLD = float(os.getenv("WHISPER_LOGPROB_THOLD", "-0.8"))
DEFAULT_WHISPER_NO_SPEECH_THOLD = float(os.getenv("WHISPER_NO_SPEECH_THOLD", "0.5"))
DEFAULT_WHISPER_VAD = os.getenv("WHISPER_VAD", "0").strip().lower() in {"1", "true", "yes", "on"}
DEFAULT_WHISPER_VAD_MODEL = os.getenv("WHISPER_VAD_MODEL", "").strip()
DEFAULT_ASR_LOOP_STREAK_THOLD = int(os.getenv("ASR_LOOP_STREAK_THOLD", "10"))
DEFAULT_ASR_BRACKET_LOOP_STREAK_THOLD = int(os.getenv("ASR_BRACKET_LOOP_STREAK_THOLD", "6"))
DEFAULT_ASR_CHUNK_FALLBACK = os.getenv("ASR_CHUNK_FALLBACK", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
DEFAULT_ASR_CHUNK_SEC = float(os.getenv("ASR_CHUNK_SEC", "120"))

_WHISPER_SUPPORTED_FLAGS: Optional[set[str]] = None


def _resolve_whisper_bin(whisper_dir: Path) -> Path:
    candidates = [
        whisper_dir / "build" / "bin" / "whisper-cli",
        whisper_dir / "build" / "bin" / "main",
        whisper_dir / "main",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"whisper.cpp binary not found under {whisper_dir}")


def _ensure_whisper_model(model_path: Path, whisper_dir: Path) -> Path:
    if model_path.exists() and model_path.stat().st_size > 0:
        return model_path

    model_path.parent.mkdir(parents=True, exist_ok=True)

    download_script = whisper_dir / "models" / "download-ggml-model.sh"
    if not download_script.exists():
        raise FileNotFoundError(f"whisper.cpp model download script not found at {download_script}")

    model_name = model_path.name.removeprefix("ggml-").removesuffix(".bin")
    subprocess.run(
        [str(download_script), model_name, str(model_path.parent)],
        check=True,
        cwd=whisper_dir,
    )

    if not model_path.exists():
        raise FileNotFoundError(f"Whisper model not found at {model_path} after download")

    return model_path


def _build_whisper_runtime_env(whisper_dir: Path) -> Dict[str, str]:
    """Build runtime env so whisper-cli can locate its shared libraries."""
    env = os.environ.copy()
    lib_paths = set()
    build_dir = whisper_dir / "build"

    # Find actual runtime library locations instead of assuming a fixed layout.
    if build_dir.exists():
        for pattern in ("libwhisper.so*", "libggml*.so*"):
            for lib_file in build_dir.rglob(pattern):
                if lib_file.is_file():
                    lib_paths.add(str(lib_file.parent))

    # Backward-compatible fallback paths for older whisper.cpp layouts.
    legacy_candidates = [
        whisper_dir / "build" / "src",
        whisper_dir / "build" / "ggml" / "src",
        whisper_dir / "build" / "ggml" / "src" / "ggml-blas",
        whisper_dir / "build" / "ggml" / "src" / "ggml-cpu",
        whisper_dir / "build" / "ggml" / "src" / "ggml-cuda",
    ]
    for path in legacy_candidates:
        if path.exists():
            lib_paths.add(str(path))

    if not lib_paths:
        return env

    existing = env.get("LD_LIBRARY_PATH", "")
    resolved_paths = sorted(lib_paths)
    env["LD_LIBRARY_PATH"] = ":".join(resolved_paths + ([existing] if existing else []))
    return env


def _max_consecutive_repeat(segments: List[Dict]) -> Tuple[int, str]:
    max_streak = 0
    max_phrase = ""
    curr_streak = 0
    curr_phrase = ""

    for seg in segments:
        phrase = seg.get("norm_text", "")
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
    tail_segments = [s for s in segments if s["start_sec"] >= start_cutoff and s.get("norm_text")]

    if not tail_segments:
        return 0, "", 0.0, 0.0

    counts = Counter(s["norm_text"] for s in tail_segments)
    top_phrase, top_count = counts.most_common(1)[0]
    ratio = top_count / len(tail_segments)
    tail_duration = tail_segments[-1]["end_sec"] - tail_segments[0]["start_sec"]
    return top_count, top_phrase, ratio, max(0.0, tail_duration)


def _is_looping_transcript(segments: List[Dict]) -> Tuple[bool, List[str]]:
    """Detect likely ASR degeneration loops (repeated phrase dominance)."""
    if not segments:
        return True, ["empty_transcript"]

    reasons = []
    max_streak, streak_phrase = _max_consecutive_repeat(segments)
    if max_streak >= DEFAULT_ASR_LOOP_STREAK_THOLD:
        reasons.append(f"long_consecutive_repeat:{max_streak}:{streak_phrase}")

    # Bracketed narrator/caption repeats (e.g. [This is a video of...]) are a common
    # early signal of whisper degeneration and should trigger sooner.
    bracket_like = streak_phrase.strip().startswith("[") and streak_phrase.strip().endswith("]")
    if bracket_like and max_streak >= DEFAULT_ASR_BRACKET_LOOP_STREAK_THOLD:
        reasons.append(f"bracket_repeat:{max_streak}:{streak_phrase}")

    tail_count, tail_phrase, tail_ratio, tail_duration = _tail_repeat_stats(segments, tail_window_sec=900.0)
    if tail_count >= 40 and tail_ratio >= 0.65 and tail_duration >= 300.0:
        reasons.append(
            f"tail_repeat_dominance:count={tail_count}:ratio={tail_ratio:.3f}:dur={tail_duration:.1f}:phrase={tail_phrase}"
        )

    return len(reasons) > 0, reasons


def _model_path_for_name(model_name: str, model_dir: Path) -> Path:
    name = model_name.strip()
    if name.endswith(".bin"):
        return model_dir / name
    if name.startswith("ggml-"):
        return model_dir / f"{name}.bin"
    return model_dir / f"ggml-{name}.bin"


def _candidate_model_paths(primary_model: Path) -> List[Path]:
    candidates = [primary_model]
    retry_model = _model_path_for_name(DEFAULT_WHISPER_RETRY_MODEL, primary_model.parent)
    if retry_model != primary_model:
        candidates.append(retry_model)
    return candidates


def _discover_supported_whisper_flags(whisper_bin: Path, whisper_dir: Path) -> set[str]:
    global _WHISPER_SUPPORTED_FLAGS
    if _WHISPER_SUPPORTED_FLAGS is not None:
        return _WHISPER_SUPPORTED_FLAGS

    try:
        result = subprocess.run(
            [str(whisper_bin), "--help"],
            check=True,
            capture_output=True,
            text=True,
            env=_build_whisper_runtime_env(whisper_dir),
        )
        help_text = f"{result.stdout}\n{result.stderr}"
    except Exception:
        help_text = ""

    supported = set()
    for flag in (
        "--temperature",
        "--temperature-inc",
        "--entropy-thold",
        "--logprob-thold",
        "--no-speech-thold",
        "--vad",
        "--vad-model",
    ):
        if flag in help_text:
            supported.add(flag)

    _WHISPER_SUPPORTED_FLAGS = supported
    return supported


def _build_decode_args(whisper_bin: Path, whisper_dir: Path, attempt_idx: int) -> List[str]:
    supported = _discover_supported_whisper_flags(whisper_bin, whisper_dir)
    args: List[str] = []
    attempt_temperature = min(
        1.0,
        DEFAULT_WHISPER_TEMPERATURE + (attempt_idx * DEFAULT_WHISPER_RETRY_TEMPERATURE_BUMP),
    )

    if "--temperature" in supported:
        args += ["--temperature", str(attempt_temperature)]
    if "--temperature-inc" in supported:
        args += ["--temperature-inc", str(DEFAULT_WHISPER_TEMPERATURE_INC)]
    if "--entropy-thold" in supported:
        args += ["--entropy-thold", str(DEFAULT_WHISPER_ENTROPY_THOLD)]
    if "--logprob-thold" in supported:
        args += ["--logprob-thold", str(DEFAULT_WHISPER_LOGPROB_THOLD)]
    if "--no-speech-thold" in supported:
        args += ["--no-speech-thold", str(DEFAULT_WHISPER_NO_SPEECH_THOLD)]

    if DEFAULT_WHISPER_VAD and "--vad" in supported:
        args += ["--vad"]
        if DEFAULT_WHISPER_VAD_MODEL and "--vad-model" in supported:
            vad_model_path = Path(DEFAULT_WHISPER_VAD_MODEL)
            if vad_model_path.exists():
                args += ["--vad-model", str(vad_model_path)]
            else:
                print(
                    f"WHISPER_VAD_MODEL path not found ({vad_model_path}); continuing without --vad-model"
                )

    return args


def _run_whisper_once(
    whisper_bin: Path,
    whisper_model: Path,
    wav_path: Path,
    out_prefix: Path,
    threads: int,
    whisper_dir: Path,
    decode_args: List[str],
) -> Path:
    json_path = Path(f"{out_prefix}.json")
    if json_path.exists():
        json_path.unlink()

    cmd = [
        str(whisper_bin),
        "-m",
        str(whisper_model),
        "-f",
        str(wav_path),
        "-t",
        str(threads),
        "-oj",
        "-of",
        str(out_prefix),
        "-np",
    ]
    cmd.extend(decode_args)
    subprocess.run(cmd, check=True, env=_build_whisper_runtime_env(whisper_dir))

    if not json_path.exists():
        raise RuntimeError(f"Whisper JSON output expected at {json_path} but not found.")
    return json_path


def _decode_args_without_vad(decode_args: List[str]) -> List[str]:
    """Strip VAD flags for fallback retry if VAD/model compatibility fails."""
    stripped: List[str] = []
    skip_next = False
    for idx, arg in enumerate(decode_args):
        if skip_next:
            skip_next = False
            continue
        if arg == "--vad":
            continue
        if arg == "--vad-model":
            # Also drop the following model-path token when present.
            if idx + 1 < len(decode_args):
                skip_next = True
            continue
        stripped.append(arg)
    return stripped


def _wav_duration_seconds(wav_path: Path) -> float:
    with wave.open(str(wav_path), "rb") as wav_f:
        frames = wav_f.getnframes()
        rate = wav_f.getframerate()
        if rate <= 0:
            return 0.0
        return frames / float(rate)


def _extract_wav_chunk(source_wav: Path, chunk_wav: Path, start_sec: float, duration_sec: float) -> None:
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        str(start_sec),
        "-t",
        str(duration_sec),
        "-i",
        str(source_wav),
        str(chunk_wav),
    ]
    subprocess.run(cmd, check=True)


def _offset_transcripts(transcripts: List[Dict], offset_sec: float) -> List[Dict]:
    adjusted = []
    for item in transcripts:
        adjusted.append(
            {
                "start_sec": round(item["start_sec"] + offset_sec, 2),
                "end_sec": round(item["end_sec"] + offset_sec, 2),
                "text": item["text"],
                "norm_text": item.get("norm_text", _normalize_text(item["text"])),
            }
        )
    return adjusted


def _run_whisper_asr_chunked(
    wav_path: Path,
    threads: int,
    whisper_bin: Path,
    whisper_dir: Path,
    model_candidates: List[Path],
    max_retries: int,
    chunk_sec: float,
) -> List[Dict]:
    duration = _wav_duration_seconds(wav_path)
    if duration <= 0:
        raise RuntimeError(f"Unable to determine WAV duration for chunk fallback: {wav_path}")

    chunk_dir = wav_path.parent / "asr_chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    merged: List[Dict] = []
    num_chunks = int((duration + chunk_sec - 1) // chunk_sec)

    print(
        f"Falling back to chunked ASR for {wav_path.name}: "
        f"duration={duration/60.0:.1f}m, chunk={chunk_sec:.0f}s, chunks={num_chunks}"
    )

    for chunk_idx in range(num_chunks):
        start_sec = chunk_idx * chunk_sec
        dur = min(chunk_sec, max(0.0, duration - start_sec))
        if dur <= 0:
            continue

        chunk_wav = chunk_dir / f"{wav_path.stem}.chunk_{chunk_idx:04d}.wav"
        _extract_wav_chunk(wav_path, chunk_wav, start_sec=start_sec, duration_sec=dur)

        chunk_out_prefix = chunk_dir / f"{wav_path.stem}.chunk_{chunk_idx:04d}"
        chunk_attempts = max_retries + 1
        chunk_ok = False
        chunk_last_error = None

        for attempt_idx in range(chunk_attempts):
            model_idx = min(attempt_idx, len(model_candidates) - 1)
            model_path = _ensure_whisper_model(model_candidates[model_idx], whisper_dir)
            decode_args = _build_decode_args(whisper_bin, whisper_dir, attempt_idx=attempt_idx)

            try:
                try:
                    chunk_json = _run_whisper_once(
                        whisper_bin=whisper_bin,
                        whisper_model=model_path,
                        wav_path=chunk_wav,
                        out_prefix=chunk_out_prefix,
                        threads=threads,
                        whisper_dir=whisper_dir,
                        decode_args=decode_args,
                    )
                except subprocess.CalledProcessError:
                    vad_enabled = "--vad" in decode_args or "--vad-model" in decode_args
                    if not vad_enabled:
                        raise
                    chunk_json = _run_whisper_once(
                        whisper_bin=whisper_bin,
                        whisper_model=model_path,
                        wav_path=chunk_wav,
                        out_prefix=chunk_out_prefix,
                        threads=threads,
                        whisper_dir=whisper_dir,
                        decode_args=_decode_args_without_vad(decode_args),
                    )

                chunk_transcripts = _parse_whisper_json(chunk_json)
                looped, reasons = _is_looping_transcript(chunk_transcripts)
                if looped:
                    chunk_last_error = RuntimeError(
                        f"chunk {chunk_idx} looped: {', '.join(reasons)}"
                    )
                    if attempt_idx < chunk_attempts - 1:
                        continue
                    raise chunk_last_error

                merged.extend(_offset_transcripts(chunk_transcripts, offset_sec=start_sec))
                chunk_ok = True
                break
            except Exception as exc:
                chunk_last_error = exc
                if attempt_idx < chunk_attempts - 1:
                    continue

        if not chunk_ok:
            print(f"Skipping ASR chunk {chunk_idx + 1}/{num_chunks} after failures: {chunk_last_error}")

        # best effort cleanup for chunk wav/json artifacts
        for ext in (".wav", ".json", ".txt", ".srt", ".vtt"):
            p = chunk_dir / f"{wav_path.stem}.chunk_{chunk_idx:04d}{ext}"
            if p.exists():
                try:
                    p.unlink()
                except OSError:
                    pass

    if not merged:
        raise RuntimeError("Chunked ASR fallback produced no transcripts")

    merged.sort(key=lambda s: (s["start_sec"], s["end_sec"]))
    final_looped, final_reasons = _is_looping_transcript(merged)
    if final_looped:
        raise RuntimeError("Chunked ASR fallback still looped: " + ", ".join(final_reasons))

    return merged


def run_whisper_asr(
    wav_path: Path,
    threads: Optional[int] = None,
    max_retries: Optional[int] = None,
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
    whisper_dir = DEFAULT_WHISPER_DIR
    whisper_bin = _resolve_whisper_bin(whisper_dir)
    whisper_model = DEFAULT_WHISPER_MODEL
    if threads is None:
        threads = int(os.getenv("WHISPER_THREADS", "4"))
    if max_retries is None:
        max_retries = DEFAULT_WHISPER_MAX_RETRIES
    if not wav_path.exists():
        raise FileNotFoundError(f"WAV audio file not found at {wav_path}")

    output_dir = wav_path.parent / "asr"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_prefix = output_dir / wav_path.stem
    json_path = Path(f"{out_prefix}.json")

    if json_path.exists() and json_path.stat().st_size > 0:
        cached = _parse_whisper_json(json_path)
        looped, reasons = _is_looping_transcript(cached)
        if not looped:
            return cached
        print(f"Cached ASR for {wav_path.name} looks degenerate ({'; '.join(reasons)}); retrying...")

    model_candidates = _candidate_model_paths(whisper_model)
    attempts = max_retries + 1
    last_error = None
    for attempt_idx in range(attempts):
        model_idx = min(attempt_idx, len(model_candidates) - 1)
        model_path = _ensure_whisper_model(model_candidates[model_idx], whisper_dir)
        decode_args = _build_decode_args(whisper_bin, whisper_dir, attempt_idx=attempt_idx)
        temp_display = min(
            1.0,
            DEFAULT_WHISPER_TEMPERATURE + (attempt_idx * DEFAULT_WHISPER_RETRY_TEMPERATURE_BUMP),
        )
        print(
            f"Running ASR on {wav_path.name} via whisper.cpp "
            f"(attempt {attempt_idx + 1}/{attempts}, model={model_path.name}, "
            f"threads={threads}, temp={temp_display})..."
        )

        try:
            try:
                produced_json = _run_whisper_once(
                    whisper_bin=whisper_bin,
                    whisper_model=model_path,
                    wav_path=wav_path,
                    out_prefix=out_prefix,
                    threads=threads,
                    whisper_dir=whisper_dir,
                    decode_args=decode_args,
                )
            except subprocess.CalledProcessError as vad_exc:
                vad_enabled = "--vad" in decode_args or "--vad-model" in decode_args
                if not vad_enabled:
                    raise

                # Some whisper.cpp builds reject certain VAD model formats/paths.
                # Fall back to the same decode attempt with VAD disabled.
                print(
                    f"VAD decode failed for {wav_path.name} (exit={vad_exc.returncode}); "
                    "retrying this attempt without VAD flags"
                )
                produced_json = _run_whisper_once(
                    whisper_bin=whisper_bin,
                    whisper_model=model_path,
                    wav_path=wav_path,
                    out_prefix=out_prefix,
                    threads=threads,
                    whisper_dir=whisper_dir,
                    decode_args=_decode_args_without_vad(decode_args),
                )

            transcripts = _parse_whisper_json(produced_json)
            looped, reasons = _is_looping_transcript(transcripts)
            if not looped:
                return transcripts

            last_error = RuntimeError(
                "ASR transcript flagged as likely loop failure: " + ", ".join(reasons)
            )
            if attempt_idx < attempts - 1:
                print(f"Retrying ASR for {wav_path.name} after loop detection: {', '.join(reasons)}")
        except Exception as exc:
            last_error = exc
            if attempt_idx < attempts - 1:
                print(f"Retrying ASR for {wav_path.name} after failure: {exc}")

    if DEFAULT_ASR_CHUNK_FALLBACK and DEFAULT_ASR_CHUNK_SEC > 0:
        return _run_whisper_asr_chunked(
            wav_path=wav_path,
            threads=threads,
            whisper_bin=whisper_bin,
            whisper_dir=whisper_dir,
            model_candidates=model_candidates,
            max_retries=max_retries,
            chunk_sec=DEFAULT_ASR_CHUNK_SEC,
        )

    raise RuntimeError(f"ASR failed after {attempts} attempt(s): {last_error}")


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
                "norm_text": _normalize_text(text),
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


def _normalize_text(text: str) -> str:
    lowered = text.lower().strip()
    cleaned = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in lowered)
    return " ".join(cleaned.split())


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
