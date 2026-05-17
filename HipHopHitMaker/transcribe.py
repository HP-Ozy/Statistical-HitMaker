#!/usr/bin/env python3
"""
transcribe.py
─────────────────────────────────────────────────────────────────────────────
Sistema di riconoscimento parole da file audio.
Modello: nvidia/canary-qwen-2.5b  (NVIDIA NeMo SALM)

ARCHITETTURA:
  [File Audio] → [C Library: libsndfile/ffmpeg] → [PCM float32 grezzo]
              → [Python: scipy resample]         → [16kHz mono float32]
              → [Canary-Qwen SALM (NeMo)]        → [Testo trascritto]
              → [LLM Mode opzionale]             → [Riassunto / Q&A]

TEORIA APPLICATA:
  1. CAMPIONAMENTO: audio digitale = valori float32 campionati nel tempo
  2. RESAMPLING: scipy.signal.resample() → FFT-based, porta a 16kHz
  3. FAST CONFORMER: encoder audio con attention lineare O(n) invece di O(n²)
     → processa fino a 40s di audio in un solo chunk senza problemi
  4. SALM (Speech-Augmented Language Model):
     encoder FastConformer: audio → embedding
     decoder Qwen3-1.7B:   embedding → testo (autoregressive)
     cross-attention:       il decoder "guarda" l'audio ad ogni token
  5. DUAL MODE:
     ASR mode  → capisce suono grezzo, trascrive con punteggiatura + maiuscole
     LLM mode  → usa solo il testo trascritto, ragiona, risponde, riassume

PERCHÉ CANARY > WHISPER (per questa applicazione):
  - WER 5.63% vs ~9% Whisper large-v3 (Open ASR Leaderboard)
  - 418 RTFx (418× real-time) vs ~1× Whisper large
  - Punteggiatura e maiuscole native
  - LLM mode: post-processing nello stesso modello, zero overhead

USO:
  python transcribe.py file.wav
  python transcribe.py file1.mp3 file2.flac
  python transcribe.py *.wav --llm-prompt "Riassumi in 3 punti"
  python transcribe.py meeting.wav --llm-prompt "Quali parole tecniche compaiono?"

FORMATI SUPPORTATI:
  WAV, FLAC, OGG, AIFF, AU  (via libsndfile  — C library)
  MP3, AAC, M4A, WMA, OPUS  (via ffmpeg pipe — C library)
─────────────────────────────────────────────────────────────────────────────
"""

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
import soundfile as sf   # scrive WAV temporanei (input a NeMo)
import torch

# NeMo SALM — Speech-Augmented Language Model
# speechlm2 è il sottomodulo NeMo per modelli ibridi audio+LLM
from nemo.collections.speechlm2.models import SALM


# ─────────────────────────────────────────────────────────────────────────────
# 1. BRIDGE CTYPES → C LIBRARY
#    Invariato rispetto alla versione Whisper.
#    Canary-Qwen vuole lo stesso input: 16kHz mono float32.
# ─────────────────────────────────────────────────────────────────────────────

class AudioBuffer(ctypes.Structure):
    """
    Specchio esatto della struct AudioBuffer in audio_preprocessor.c
    L'ordine e i tipi dei campi DEVONO coincidere con il codice C.
    """
    _fields_ = [
        ("samples",     ctypes.POINTER(ctypes.c_float)),
        ("num_samples", ctypes.c_int),
        ("sample_rate", ctypes.c_int),
        ("channels",    ctypes.c_int),
        ("error_code",  ctypes.c_int),
        ("error_msg",   ctypes.c_char * 256),
    ]


def load_c_library() -> ctypes.CDLL:
    script_dir = Path(__file__).parent
    lib_path   = script_dir / "libaudio.so"

    if not lib_path.exists():
        print("libaudio.so non trovata! Esegui: make")
        sys.exit(1)

    lib = ctypes.CDLL(str(lib_path))
    lib.load_audio_file.argtypes  = [ctypes.c_char_p]
    lib.load_audio_file.restype   = ctypes.POINTER(AudioBuffer)
    lib.free_audio_buffer.argtypes = [ctypes.POINTER(AudioBuffer)]
    lib.free_audio_buffer.restype  = None
    lib.get_format_info.argtypes   = [
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_char_p,
        ctypes.c_int,
    ]
    lib.get_format_info.restype = None
    return lib


# ─────────────────────────────────────────────────────────────────────────────
# 2. LETTURA AUDIO TRAMITE LIBRERIA C
# ─────────────────────────────────────────────────────────────────────────────

def read_audio_via_c(lib: ctypes.CDLL, filepath: str) -> dict:
    c_path  = filepath.encode("utf-8")
    sr_out  = ctypes.c_int(0)
    ch_out  = ctypes.c_int(0)
    fr_out  = ctypes.c_int(0)
    fmt_buf = ctypes.create_string_buffer(64)

    lib.get_format_info(c_path,
                        ctypes.byref(sr_out), ctypes.byref(ch_out),
                        ctypes.byref(fr_out), fmt_buf, 64)

    fmt_str = fmt_buf.value.decode("utf-8")
    buf_ptr = lib.load_audio_file(c_path)

    if not buf_ptr:
        raise RuntimeError(f"Puntatore null dalla C library per: {filepath}")

    buf = buf_ptr.contents
    if buf.error_code != 0:
        msg = buf.error_msg.decode("utf-8")
        lib.free_audio_buffer(buf_ptr)
        raise RuntimeError(f"C library error {buf.error_code}: {msg}")

    # np.frombuffer crea una VIEW del buffer C → .copy() rende l'array indipendente
    # FONDAMENTALE: copiare PRIMA di free_audio_buffer, altrimenti segfault
    samples_np = np.frombuffer(
        (ctypes.c_float * buf.num_samples).from_address(
            ctypes.addressof(buf.samples.contents)
        ),
        dtype=np.float32,
    ).copy()

    sr, ch = buf.sample_rate, buf.channels
    lib.free_audio_buffer(buf_ptr)

    return {"samples": samples_np, "sample_rate": sr,
            "channels": ch, "format": fmt_str}


# ─────────────────────────────────────────────────────────────────────────────
# 3. PREPROCESSING AUDIO
#    Canary-Qwen richiede: float32, mono, 16000 Hz
#    Il frontend audio interno (log-mel 80 bins) è gestito da NeMo.
# ─────────────────────────────────────────────────────────────────────────────

TARGET_SR = 16000  # Hz

def preprocess_audio(samples: np.ndarray, sample_rate: int, channels: int) -> np.ndarray:
    """
    1. Multi-canale → Mono (media aritmetica, layout interleaved)
    2. Resample → 16000 Hz (FFT-based via scipy)
    3. Clip in [-1.0, +1.0]
    """
    if channels > 1:
        # Layout interleaved: [L0, R0, L1, R1, ...] → reshape → media
        samples = samples.reshape(-1, channels).mean(axis=1)

    if sample_rate != TARGET_SR:
        new_len = int(len(samples) * TARGET_SR / sample_rate)
        samples = scipy.signal.resample(samples, new_len).astype(np.float32)

    max_val = np.abs(samples).max()
    if max_val > 1.0:
        samples = samples / max_val

    return samples.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 4. CARICAMENTO MODELLO CANARY-QWEN
# ─────────────────────────────────────────────────────────────────────────────

def load_canary_model(device: str = "auto") -> SALM:
    """
    Carica nvidia/canary-qwen-2.5b tramite NeMo.

    SALM.from_pretrained() scarica e mette in cache i pesi da HuggingFace.
    Cache locale: ~/.cache/huggingface/hub/

    ARCHITETTURA del modello:
    - Encoder:  FastConformer (da nvidia/canary-1b-flash, 34 layer)
    - Decoder:  Qwen3-1.7B con LoRA adapters per la parte audio
    - Tokenizer: Qwen3 tokenizer
    - Totale:   ~2.5B parametri
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"   Device: {device.upper()}")
    model = SALM.from_pretrained("nvidia/canary-qwen-2.5b")
    model.eval()  # disattiva dropout e batch norm training mode

    if device == "cuda":
        model = model.cuda()

    return model


# ─────────────────────────────────────────────────────────────────────────────
# 5. TRASCRIZIONE ASR MODE
#
#    FLUSSO INTERNO CANARY-QWEN (ASR mode):
#    ┌───────────────────────────────────────────────────────────────┐
#    │  PCM 16kHz  →  Log-Mel Spectrogram (80 mel bins, STFT)       │
#    │             →  FastConformer Encoder (CNN locale + Attn glob) │
#    │                 ↓  audio embeddings  [T_audio, d_model]       │
#    │  "<|audioplaceholder|>"  →  Qwen3 Tokenizer                  │
#    │  Cross-Attention: token decoder guarda audio embeddings       │
#    │  Autoregressive decode: genera un token alla volta            │
#    │                 ↓                                             │
#    │  Testo con punteggiatura, maiuscole, speaker disfluencies     │
#    └───────────────────────────────────────────────────────────────┘
# ─────────────────────────────────────────────────────────────────────────────

def transcribe_asr(model: SALM, samples: np.ndarray) -> str:
    """
    Trascrive i campioni audio in testo.

    Approccio: salva WAV temporaneo → model.transcribe([path])
    NeMo gestisce internamente PCM → log-mel → embedding → testo.
    """
    # Scrivi WAV temporaneo (NeMo vuole path su disco o DataLoader)
    fd, tmp_path = tempfile.mkstemp(suffix=".wav", prefix="canary_")
    os.close(fd)

    try:
        sf.write(tmp_path, samples, TARGET_SR, subtype="FLOAT")

        results = model.transcribe(
            audio      = [tmp_path],
            batch_size = 1,
            task       = "asr",      # ASR mode: processa l'audio direttamente
            pnc        = True,       # Punctuation aNd Capitalization
            verbose    = False,
        )

        # results può essere lista di stringhe o lista di oggetti con .text
        r = results[0]
        return r if isinstance(r, str) else r.text

    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


# ─────────────────────────────────────────────────────────────────────────────
# 6. LLM MODE — FEATURE ESCLUSIVA DI CANARY-QWEN
#
#    In LLM mode il modello NON processa più l'audio.
#    Usa solo il testo della trascrizione come input dell'LLM (Qwen3).
#
#    PERCHÉ È POSSIBILE:
#    Il decoder Qwen3 è un vero LLM completo. In ASR mode riceve gli
#    embedding audio dall'encoder. In LLM mode bypassa l'encoder e
#    riceve direttamente token testuali → le sue capacità reasoning restano.
#
#    WHISPER non può fare questo: il suo decoder è addestrato SOLO per ASR,
#    non è un LLM general purpose.
# ─────────────────────────────────────────────────────────────────────────────

def query_llm_mode(model: SALM, transcript: str, user_prompt: str) -> str:
    """
    Usa il decoder Qwen3 in modalità testo pura per ragionare sul trascritto.

    Esempi di user_prompt:
    - "Riassumi il contenuto in 3 punti"
    - "Quali termini tecnici compaiono?"
    - "Di cosa parla questo audio?"
    - "Elenca i nomi propri menzionati"
    """
    messages = [
        {
            "role": "system",
            "content": "Sei un assistente che analizza trascrizioni audio."
        },
        {
            "role": "user",
            "content": f"Trascrizione audio:\n\n{transcript}\n\n{user_prompt}"
        },
    ]

    # llm_generate bypassa l'encoder audio e usa solo il decoder Qwen3
    try:
        output = model.llm_generate(
            prompts        = [messages],
            max_new_tokens = 512,
            temperature    = 0.7,
        )
    except AttributeError:
        # Fallback su API generica se llm_generate non è disponibile
        output = model.generate(
            prompts        = [messages],
            max_new_tokens = 512,
        )

    if isinstance(output, (list, tuple)):
        return str(output[0]) if output else ""
    return str(output)


# ─────────────────────────────────────────────────────────────────────────────
# 7. ESTRAZIONE PAROLE CON TIMESTAMP STIMATI
#
#    Canary-Qwen non produce word-level timestamps nativi.
#    Strategia: distribuzione lineare nel tempo sulla durata dell'audio.
#
#    Per timestamp precisi in futuro:
#    → ctc_segmentation (allineamento CTC sul testo trascritto)
#    → nemo.collections.asr.parts.utils.streaming_utils (forced alignment)
# ─────────────────────────────────────────────────────────────────────────────

_PUNCT_STRIP = re.compile(r"^[^\w']+|[^\w']+$")

def extract_words(text: str, duration_sec: float) -> list[dict]:
    """
    Divide il testo in parole e assegna timestamp stimati lineari.
    """
    raw_tokens = text.split()
    words = [_PUNCT_STRIP.sub("", t) for t in raw_tokens]
    words = [w for w in words if w]

    if not words:
        return []

    word_dur = duration_sec / len(words)
    return [
        {
            "word":  w,
            "start": round(i * word_dur, 2),
            "end":   round((i + 1) * word_dur, 2),
        }
        for i, w in enumerate(words)
    ]


# ─────────────────────────────────────────────────────────────────────────────
# 8. OUTPUT FORMATTATO
# ─────────────────────────────────────────────────────────────────────────────

SEP = "═" * 68

def print_header():
    print(f"\n{SEP}")
    print("  AUDIO WORD RECOGNIZER  —  C + Python + nvidia/canary-qwen-2.5b")
    print(SEP)


def print_file_result(filepath, audio_info, duration_sec,
                      word_list, full_text, llm_output, proc_time):
    filename = Path(filepath).name
    print(f"\n FILE: {filename}")
    print(f"   Formato:    {audio_info['format']} | "
          f"{audio_info['channels']}ch | "
          f"{audio_info['sample_rate']} Hz | "
          f"{duration_sec:.1f}s | proc {proc_time:.2f}s")

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
    if unique:
        print(f"\n  VOCABOLARIO UNICO:")
        cols = 5
        for i in range(0, len(unique), cols):
            print("  " + "  ".join(f"{w:<16}" for w in unique[i:i+cols]))
    print(f"{SEP}\n")


# ─────────────────────────────────────────────────────────────────────────────
# 9. MAIN
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Audio word recognizer — Canary-Qwen-2.5B")
    p.add_argument("files", nargs="+",
                   help="File audio: wav, flac, ogg, aiff, mp3, aac, m4a, wma, opus")
    p.add_argument("--llm-prompt", default=None,
                   help="LLM mode: prompt per ragionare sul testo trascritto "
                        "(es: 'Riassumi in 3 punti')")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    return p.parse_args()


def main():
    args = parse_args()

    # Carica libreria C
    print("\n Caricamento libreria C...")
    lib = load_c_library()
    print("  libaudio.so OK")

    # Carica Canary-Qwen
    print(f"\n Caricamento nvidia/canary-qwen-2.5b ...")
    print("  (prima esecuzione: download ~5GB da HuggingFace)")
    model = load_canary_model(args.device)
    print("  Canary-Qwen-2.5B OK")

    if args.llm_prompt:
        print(f"  LLM mode attivo: \"{args.llm_prompt}\"")

    print_header()

    all_words, successful = [], 0

    for filepath in args.files:
        if not os.path.exists(filepath):
            print(f"\n  File non trovato: {filepath}")
            continue

        t0 = time.time()
        try:
            print(f"\n Elaborazione: {Path(filepath).name} ...")

            # 1. Leggi audio via C
            audio_info   = read_audio_via_c(lib, filepath)
            duration_sec = len(audio_info["samples"]) / (
                audio_info["sample_rate"] * max(audio_info["channels"], 1)
            )

            # 2. Preprocessing
            samples = preprocess_audio(
                audio_info["samples"],
                audio_info["sample_rate"],
                audio_info["channels"],
            )

            # 3. Trascrizione ASR
            transcript = transcribe_asr(model, samples)

            # 4. LLM mode (opzionale)
            llm_output = None
            if args.llm_prompt and transcript.strip():
                print("    LLM mode...")
                llm_output = query_llm_mode(model, transcript, args.llm_prompt)

            # 5. Estrai parole + timestamp stimati
            word_list = extract_words(transcript, duration_sec)
            all_words.extend(w["word"] for w in word_list)

            print_file_result(
                filepath, audio_info, duration_sec,
                word_list, transcript, llm_output,
                time.time() - t0,
            )
            successful += 1

        except Exception as e:
            print(f"\n  Errore su {filepath}: {e}")
            import traceback; traceback.print_exc()

    print_summary(len(args.files), successful, all_words)


if __name__ == "__main__":
    main()
