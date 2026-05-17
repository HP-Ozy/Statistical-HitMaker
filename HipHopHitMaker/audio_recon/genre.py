#!/usr/bin/env python3
"""
genre.py
─────────────────────────────────────────────────────────────────────────────
Riconoscimento del GENERE MUSICALE — ramo parallelo all'ASR.

PERCHÉ UN MODULO SEPARATO (non Canary-Qwen):
  L'encoder FastConformer di Canary è addestrato per essere INVARIANTE a
  timbro/strumentazione/ritmo: butta via di proposito l'informazione che
  serve al genere. In LLM mode il decoder vede solo il TESTO trascritto.
  → Una barra trap e una ballata country con lo stesso testo sono identiche
    per Canary. Il genere richiede un modello che "ascolti la musica",
    non che "trascriva le parole".

ARCHITETTURA (fork del buffer PCM):

  ┌──────────────────────────────────────────────────────────────────────┐
  │  [C lib: PCM float32 grezzo]  (read_audio_via_c — INVARIATO)         │
  │            │                                                         │
  │            ├─→ preprocess 16kHz → Canary-Qwen → testo/parole         │
  │            │                                                         │
  │            └─→ resample @ SR_modello                                 │
  │                 → sliding window (finestre da ~Ns, hop Hs)           │
  │                 → modello audio-classification (HF transformers)     │
  │                 → softmax per finestra → MEDIA sui softmax           │
  │                 → argmax → genere + confidenza + top-k               │
  └──────────────────────────────────────────────────────────────────────┘

TEORIA — perché sliding window + media:
  Un brano dura 3-4 min. I modelli MGC vogliono finestre da ~30s.
  L'intro è silenziosa, il drop è pesante, il bridge cambia tessitura.
  Una singola finestra MENTE. Si segmenta il brano, si classifica ogni
  finestra, si MEDIANO i softmax (non gli argmax: la media dei voti
  perde l'incertezza, la media delle probabilità la conserva).
  Questa è l'aggregazione chunk-level standard in letteratura MGC.

MODELLI COMPATIBILI (qualsiasi checkpoint HF "audio-classification"):
  - AST  (MIT/ast-finetuned-*)        — ~85% su GTZAN, transformer su mel
  - wav2vec2 fine-tuned music genre    — torch puro, leggero
  Passa l'id via GenreClassifier(model_id="...").
  GTZAN ha 10 generi: blues classical country disco hiphop jazz metal
  pop reggae rock. ATTENZIONE: GTZAN è un dataset-giocattolo (Sturm 2013:
  tracce duplicate/mislabelate). Ottimo per demo, debole per produzione.
  Per produzione → Essentia discogs-effnet (400 stili) via ONNX.

DIPENDENZE: torch, transformers, scipy  (già nel tuo env NeMo + scipy)
─────────────────────────────────────────────────────────────────────────────
"""

import numpy as np
import scipy.signal
import torch


# ─────────────────────────────────────────────────────────────────────────────
# CLASSIFICATORE GENERE
# ─────────────────────────────────────────────────────────────────────────────

class GenreClassifier:
    """
    Wrapper su un modello HF di audio-classification.

    Uso:
        gc = GenreClassifier()                 # carica il modello una volta
        result = gc.predict(samples, sr, ch)   # samples = output C lib GREZZO

    `samples`/`sr`/`ch` sono ESATTAMENTE i valori di read_audio_via_c():
    PCM float32 interleaved, sample rate ORIGINALE, n. canali ORIGINALE.
    Il resampling al SR del modello è interno (NON riusare i 16kHz
    dell'ASR: ogni modello MGC ha il suo SR atteso).
    """

    def __init__(
        self,
        model_id: str = "dima806/music_genres_classification",
        device: str = "auto",
        window_sec: float = 10.0,
        hop_sec: float = 5.0,
    ):
        # Import locale: se non usi il genere, non paghi l'import di transformers
        from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        print(f"   Caricamento genre model: {model_id}  (device: {device.upper()})")
        self.fe = AutoFeatureExtractor.from_pretrained(model_id)
        self.model = AutoModelForAudioClassification.from_pretrained(model_id)
        self.model.eval()
        if device == "cuda":
            self.model = self.model.cuda()

        # SR richiesto dal feature extractor del modello (NON 16kHz fisso)
        self.target_sr = int(self.fe.sampling_rate)
        # id → etichetta (es. 0 → "hiphop")
        self.id2label = self.model.config.id2label

        self.window_samples = int(window_sec * self.target_sr)
        self.hop_samples = int(hop_sec * self.target_sr)

    # ── preprocessing: stesso pattern del tuo preprocess_audio, SR diverso ──
    def _prep(self, samples: np.ndarray, sample_rate: int, channels: int) -> np.ndarray:
        """Multi-canale → mono, resample → SR del modello, clip [-1, 1]."""
        if channels > 1:
            samples = samples.reshape(-1, channels).mean(axis=1)

        if sample_rate != self.target_sr:
            new_len = int(len(samples) * self.target_sr / sample_rate)
            samples = scipy.signal.resample(samples, new_len)

        samples = samples.astype(np.float32)
        peak = np.abs(samples).max()
        if peak > 1.0:
            samples = samples / peak
        return samples

    # ── segmentazione in finestre sovrapposte ──
    def _windows(self, samples: np.ndarray):
        n = len(samples)
        if n <= self.window_samples:
            yield samples
            return
        start = 0
        while start + self.window_samples <= n:
            yield samples[start : start + self.window_samples]
            start += self.hop_samples
        # coda finale (ultima parte del brano, spesso l'outro: conta comunque)
        if start < n:
            yield samples[-self.window_samples:]

    @torch.no_grad()
    def predict(
        self,
        samples: np.ndarray,
        sample_rate: int,
        channels: int,
        top_k: int = 3,
    ) -> dict:
        """
        Ritorna:
          {
            "genre":   "hiphop",
            "confidence": 0.71,
            "top_k":   [("hiphop", 0.71), ("pop", 0.14), ("rnb", 0.06)],
            "n_windows": 23,
          }
        """
        audio = self._prep(samples, sample_rate, channels)

        probs_acc = None
        n_win = 0

        for win in self._windows(audio):
            inputs = self.fe(
                win,
                sampling_rate=self.target_sr,
                return_tensors="pt",
            )
            if self.device == "cuda":
                inputs = {k: v.cuda() for k, v in inputs.items()}

            logits = self.model(**inputs).logits          # [1, n_classi]
            probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()

            # MEDIA dei softmax (conserva l'incertezza, vedi docstring)
            probs_acc = probs if probs_acc is None else probs_acc + probs
            n_win += 1

        probs_acc /= max(n_win, 1)

        order = np.argsort(probs_acc)[::-1]
        top = [
            (self.id2label[int(i)], round(float(probs_acc[i]), 4))
            for i in order[:top_k]
        ]
        best_idx = int(order[0])

        return {
            "genre": self.id2label[best_idx],
            "confidence": round(float(probs_acc[best_idx]), 4),
            "top_k": top,
            "n_windows": n_win,
        }


# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT FORMATTATO (stile coerente col tuo print_file_result)
# ─────────────────────────────────────────────────────────────────────────────

def format_genre_block(result: dict) -> str:
    """Blocco testo da inserire nel report del file."""
    if not result:
        return ""
    lines = [
        "",
        f" GENERE MUSICALE: {result['genre'].upper()}  "
        f"(confidenza {result['confidence']*100:.1f}%, "
        f"{result['n_windows']} finestre)",
        f"   {'GENERE':<16} {'PROB':>7}",
        f"   {'─'*16} {'─'*7}",
    ]
    for g, p in result["top_k"]:
        bar = "█" * max(1, round(p * 24))
        lines.append(f"   {g:<16} {p*100:>6.1f}%  {bar}")
    lines.append(
        "\n   * Genere = ramo audio dedicato (NON Canary). "
        "Aggregazione sliding-window."
    )
    return "\n".join(lines)
