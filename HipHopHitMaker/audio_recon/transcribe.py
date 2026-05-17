#!/usr/bin/env python3

import ctypes
import sys
import os
import re
import time
import tempfile
import argparse
from pathlib import Path

import numpy as np
import scipy.signal
import soundfile as sf
import torch

from nemo.collections.speechlm2.models import SALM

from genre import GenreClassifier, format_genre_block
from notes import NoteDetector, format_notes_block
from tempo import TempoDetector, format_tempo_block


class AudioBuffer(ctypes.Structure):
    _fields_ = [
        ("samples", ctypes.POINTER(ctypes.c_float)),
        ("num_samples", ctypes.c_int),
        ("sample_rate", ctypes.c_int),
        ("channels", ctypes.c_int),
        ("error_code", ctypes.c_int),
        ("error_msg", ctypes.c_char * 256),
    ]


def load_c_library() -> ctypes.CDLL:
    script_dir = Path(__file__).parent
    lib_path = script_dir / "libaudio.so"

    if not lib_path.exists():
        print("libaudio.so non trovata! Esegui: make")
        sys.exit(1)

    lib = ctypes.CDLL(str(lib_path))
    lib.load_audio_file.argtypes = [ctypes.c_char_p]
    lib.load_audio_file.restype = ctypes.POINTER(AudioBuffer)
    lib.free_audio_buffer.argtypes = [ctypes.POINTER(AudioBuffer)]
    lib.free_audio_buffer.restype = None
    lib.get_format_info.argtypes = [
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_char_p,
        ctypes.c_int,
    ]
    lib.get_format_info.restype = None

    return lib


def read_audio_via_c(lib: ctypes.CDLL, filepath: str) -> dict:
    c_path = filepath.encode("utf-8")
    sr_out = ctypes.c_int(0)
    ch_out = ctypes.c_int(0)
    fr_out = ctypes.c_int(0)
    fmt_buf = ctypes.create_string_buffer(64)

    lib.get_format_info(
        c_path,
        ctypes.byref(sr_out),
        ctypes.byref(ch_out),
        ctypes.byref(fr_out),
        fmt_buf,
        64
    )

    fmt_str = fmt_buf.value.decode("utf-8")
    buf_ptr = lib.load_audio_file(c_path)

    if not buf_ptr:
        raise RuntimeError(f"Puntatore null dalla C library per: {filepath}")

    buf = buf_ptr.contents

    if buf.error_code != 0:
        msg = buf.error_msg.decode("utf-8")
        lib.free_audio_buffer(buf_ptr)
        raise RuntimeError(f"C library error {buf.error_code}: {msg}")

    samples_np = np.frombuffer(
        (ctypes.c_float * buf.num_samples).from_address(
            ctypes.addressof(buf.samples.contents)
        ),
        dtype=np.float32,
    ).copy()

    sr, ch = buf.sample_rate, buf.channels
    lib.free_audio_buffer(buf_ptr)

    return {
        "samples": samples_np,
        "sample_rate": sr,
        "channels": ch,
        "format": fmt_str
    }


TARGET_SR = 16000


def preprocess_audio(samples: np.ndarray, sample_rate: int, channels: int) -> np.ndarray:
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)

    if sample_rate != TARGET_SR:
        new_len = int(len(samples) * TARGET_SR / sample_rate)
        samples = scipy.signal.resample(samples, new_len).astype(np.float32)

    max_val = np.abs(samples).max()

    if max_val > 1.0:
        samples = samples / max_val

    return samples.astype(np.float32)


def load_canary_model(device: str = "auto") -> SALM:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"   Device: {device.upper()}")

    model = SALM.from_pretrained("nvidia/canary-qwen-2.5b")
    model.eval()

    if device == "cuda":
        model = model.cuda()

    return model


def transcribe_asr(model: SALM, samples: np.ndarray) -> str:
    fd, tmp_path = tempfile.mkstemp(suffix=".wav", prefix="canary_")
    os.close(fd)

    try:
        sf.write(tmp_path, samples, TARGET_SR, subtype="FLOAT")

        answer_ids = model.generate(
            prompts=[
                [{
                    "role": "user",
                    "content": f"Transcribe the following: {model.audio_locator_tag}",
                    "audio": [tmp_path],
                }]
            ],
            max_new_tokens=256,
        )

        return model.tokenizer.ids_to_text(answer_ids[0].cpu())

    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def query_llm_mode(model: SALM, transcript: str, user_prompt: str) -> str:
    content = (
        "Sei un assistente che analizza trascrizioni audio.\n\n"
        f"Trascrizione audio:\n\n{transcript}\n\n{user_prompt}"
    )

    with model.llm.disable_adapter():
        answer_ids = model.generate(
            prompts=[[{"role": "user", "content": content}]],
            max_new_tokens=512,
        )

    return model.tokenizer.ids_to_text(answer_ids[0].cpu())


_PUNCT_STRIP = re.compile(r"^[^\w']+|[^\w']+$")


def extract_words(text: str, duration_sec: float) -> list[dict]:
    raw_tokens = text.split()
    words = [_PUNCT_STRIP.sub("", t) for t in raw_tokens]
    words = [w for w in words if w]

    if not words:
        return []

    word_dur = duration_sec / len(words)

    return [
        {
            "word": w,
            "start": round(i * word_dur, 2),
            "end": round((i + 1) * word_dur, 2),
        }
        for i, w in enumerate(words)
    ]


SEP = "═" * 68


def print_header():
    print(f"\n{SEP}")
    print("  AUDIO WORD RECOGNIZER  —  C + Python + nvidia/canary-qwen-2.5b")
    print(SEP)


def print_file_result(
    filepath,
    audio_info,
    duration_sec,
    word_list,
    full_text,
    llm_output,
    proc_time,
    genre_result=None,
    notes_result=None,
    tempo_result=None
):
    filename = Path(filepath).name

    print(f"\n FILE: {filename}")
    print(f"   Formato:    {audio_info['format']} | "
          f"{audio_info['channels']}ch | "
          f"{audio_info['sample_rate']} Hz | "
          f"{duration_sec:.1f}s | proc {proc_time:.2f}s")

    if genre_result:
        print(format_genre_block(genre_result))

    if notes_result:
        print(format_notes_block(notes_result))

    if tempo_result:
        print(format_tempo_block(tempo_result))

    print(f"\n TESTO COMPLETO (ASR mode):")
    print(f"   {full_text.strip()}")

    if llm_output:
        print(f"\n RISPOSTA LLM (LLM mode):")
        for line in llm_output.strip().splitlines():
            print(f"   {line}")

    print(f"\n PAROLE RICONOSCIUTE ({len(word_list)} totali):")
    print(f"   {'#':<4} {'PAROLA':<24} {'INIZIO':>8} {'FINE':>8}")
    print(f"   {'─'*4} {'─'*24} {'─'*8} {'─'*8}")

    for i, w in enumerate(word_list, 1):
        print(f"   {i:<4} {w['word']:<24} {w['start']:>7.2f}s {w['end']:>7.2f}s")

    print(f"\n   * Timestamp stimati (distribuzione lineare).")
    print(f"     Per allineamento preciso: ctc_segmentation + NeMo forced aligner.")
    print(f"\n   {'─'*55}")


def print_summary(total_files, successful, all_words):
    unique = sorted(set(w.lower() for w in all_words if w))

    print(f"\n{SEP}")
    print(f"  RIEPILOGO FINALE")
    print(SEP)
    print(f"  Modello:         nvidia/canary-qwen-2.5b  (SALM, 2.5B param, CC-BY-4.0)")
    print(f"  File processati: {successful}/{total_files}")
    print(f"  Parole totali:   {len(all_words)}")
    print(f"  Parole uniche:   {len(unique)}")

    if all_words:
        from collections import Counter

        freq = Counter(w.lower() for w in all_words if w)
        ranked = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))
        top = ranked[:20]
        max_n = top[0][1] if top else 1
        bar_w = 30

        print(f"\n  PAROLE PIÙ FREQUENTI (top {len(top)} su {len(freq)}):")
        print(f"  {'#':<3} {'PAROLA':<18} {'N':>3}  GRAFICO")
        print(f"  {'─'*3} {'─'*18} {'─'*3}  {'─'*bar_w}")

        for i, (w, n) in enumerate(top, 1):
            bar = "█" * max(1, round(n / max_n * bar_w))
            print(f"  {i:<3} {w:<18} {n:>3}  {bar}")

    if unique:
        print(f"\n  VOCABOLARIO UNICO:")
        cols = 5

        for i in range(0, len(unique), cols):
            print("  " + "  ".join(f"{w:<16}" for w in unique[i:i + cols]))

    print(f"{SEP}\n")


def parse_args():
    p = argparse.ArgumentParser(description="Audio word recognizer — Canary-Qwen-2.5B")

    p.add_argument(
        "files",
        nargs="+",
        help="File audio: wav, flac, ogg, aiff, mp3, aac, m4a, wma, opus"
    )

    p.add_argument(
        "--llm-prompt",
        default=None,
        help="LLM mode: prompt per ragionare sul testo trascritto"
    )

    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])

    p.add_argument(
        "--genre",
        action="store_true",
        help="Attiva riconoscimento genere musicale"
    )

    p.add_argument(
        "--genre-model",
        default="dima806/music_genres_classification",
        help="Checkpoint HF audio-classification per il genere"
    )

    p.add_argument(
        "--notes",
        action="store_true",
        help="Attiva analisi note più usate + tonalità"
    )

    p.add_argument(
        "--tempo",
        action="store_true",
        help="Attiva analisi BPM + tempo"
    )

    return p.parse_args()


def main():
    args = parse_args()

    print("\n Caricamento libreria C...")
    lib = load_c_library()
    print("  libaudio.so OK")

    print(f"\n Caricamento nvidia/canary-qwen-2.5b ...")
    print("  (prima esecuzione: download ~5GB da HuggingFace)")

    model = load_canary_model(args.device)
    print("  Canary-Qwen-2.5B OK")

    if args.llm_prompt:
        print(f"  LLM mode attivo: \"{args.llm_prompt}\"")

    genre_clf = None

    if args.genre:
        print(f"\n Caricamento genre model ...")
        genre_clf = GenreClassifier(model_id=args.genre_model, device=args.device)
        print("  Genre model OK")

    note_det = None

    if args.notes:
        note_det = NoteDetector()
        print("  Note detector OK")

    tempo_det = None

    if args.tempo:
        tempo_det = TempoDetector()
        print("  Tempo detector OK")

    print_header()

    all_words, successful = [], 0

    for filepath in args.files:
        if not os.path.exists(filepath):
            print(f"\n  File non trovato: {filepath}")
            continue

        t0 = time.time()

        try:
            print(f"\n Elaborazione: {Path(filepath).name} ...")

            audio_info = read_audio_via_c(lib, filepath)

            duration_sec = len(audio_info["samples"]) / (
                audio_info["sample_rate"] * max(audio_info["channels"], 1)
            )

            genre_result = None

            if genre_clf:
                print("    Genre branch...")
                genre_result = genre_clf.predict(
                    audio_info["samples"],
                    audio_info["sample_rate"],
                    audio_info["channels"],
                )

            notes_result = None

            if note_det:
                print("    Notes branch...")
                notes_result = note_det.predict(
                    audio_info["samples"],
                    audio_info["sample_rate"],
                    audio_info["channels"],
                )

            tempo_result = None

            if tempo_det:
                print("    Tempo branch...")
                tempo_result = tempo_det.predict(
                    audio_info["samples"],
                    audio_info["sample_rate"],
                    audio_info["channels"],
                )

            samples = preprocess_audio(
                audio_info["samples"],
                audio_info["sample_rate"],
                audio_info["channels"],
            )

            transcript = transcribe_asr(model, samples)

            llm_output = None

            if args.llm_prompt and transcript.strip():
                print("    LLM mode...")
                llm_output = query_llm_mode(model, transcript, args.llm_prompt)

            word_list = extract_words(transcript, duration_sec)
            all_words.extend(w["word"] for w in word_list)

            print_file_result(
                filepath,
                audio_info,
                duration_sec,
                word_list,
                transcript,
                llm_output,
                time.time() - t0,
                genre_result,
                notes_result,
                tempo_result,
            )

            successful += 1

        except Exception as e:
            print(f"\n  Errore su {filepath}: {e}")
            import traceback
            traceback.print_exc()

    print_summary(len(args.files), successful, all_words)


if __name__ == "__main__":
    main()
