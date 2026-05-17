#!/usr/bin/env python3

import numpy as np
import scipy.signal
import torch


class GenreClassifier:
    def __init__(
        self,
        model_id: str = "dima806/music_genres_classification",
        device: str = "auto",
        window_sec: float = 10.0,
        hop_sec: float = 5.0,
    ):
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

        self.target_sr = int(self.fe.sampling_rate)
        self.id2label = self.model.config.id2label

        self.window_samples = int(window_sec * self.target_sr)
        self.hop_samples = int(hop_sec * self.target_sr)

    def _prep(self, samples: np.ndarray, sample_rate: int, channels: int) -> np.ndarray:
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

    def _windows(self, samples: np.ndarray):
        n = len(samples)

        if n <= self.window_samples:
            yield samples
            return

        start = 0

        while start + self.window_samples <= n:
            yield samples[start:start + self.window_samples]
            start += self.hop_samples

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

            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()

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


def format_genre_block(result: dict) -> str:
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
