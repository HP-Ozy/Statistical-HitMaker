#!/usr/bin/env python3

import numpy as np


_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

_KS_MAJOR = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
)

_KS_MINOR = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
)


def _best_key(chroma_vec: np.ndarray) -> dict:
    v = chroma_vec - chroma_vec.mean()

    def corr(profile, shift):
        p = np.roll(profile, shift)
        p = p - p.mean()
        denom = np.linalg.norm(v) * np.linalg.norm(p)
        return float(np.dot(v, p) / denom) if denom else 0.0

    best = {"key": None, "mode": None, "score": -2.0}

    for shift in range(12):
        for mode, prof in (("major", _KS_MAJOR), ("minor", _KS_MINOR)):
            c = corr(prof, shift)

            if c > best["score"]:
                best = {
                    "key": _NOTE_NAMES[shift],
                    "mode": mode,
                    "score": c
                }

    return best


class NoteDetector:
    def __init__(self, target_sr: int = 22050):
        import librosa

        self.target_sr = target_sr

    def predict(
        self,
        samples: np.ndarray,
        sample_rate: int,
        channels: int,
        top_k: int = 5,
    ) -> dict:
        import librosa

        x = samples.astype(np.float32)

        if channels > 1:
            x = x.reshape(-1, channels).mean(axis=1)

        if sample_rate != self.target_sr:
            x = librosa.resample(
                x,
                orig_sr=sample_rate,
                target_sr=self.target_sr
            )

        x_harm = librosa.effects.harmonic(x, margin=4.0)

        chroma = librosa.feature.chroma_cqt(
            y=x_harm,
            sr=self.target_sr
        )

        energy = chroma.sum(axis=1)
        total = energy.sum()

        if total <= 0:
            return {
                "key": None,
                "mode": None,
                "key_confidence": 0.0,
                "histogram": [],
                "top_k": [],
            }

        weights = energy / total

        hist = sorted(
            ((_NOTE_NAMES[i], round(float(weights[i]), 4)) for i in range(12)),
            key=lambda kv: kv[1],
            reverse=True,
        )

        key = _best_key(energy)

        return {
            "key": key["key"],
            "mode": key["mode"],
            "key_confidence": round(key["score"], 4),
            "histogram": hist,
            "top_k": hist[:top_k],
        }


def format_notes_block(result: dict) -> str:
    if not result or not result.get("histogram"):
        return ""

    key = result["key"]
    mode = result["mode"]
    conf = result["key_confidence"]

    lines = [
        "",
        f" NOTE PIÙ USATE  —  tonalità stimata: {key} {mode}  "
        f"(corr {conf:+.2f})",
        f"   {'NOTA':<5} {'PESO':>7}",
        f"   {'─'*5} {'─'*7}",
    ]

    top = result["top_k"]
    mx = top[0][1] if top else 1.0

    for n, w in top:
        bar = "█" * max(1, round(w / mx * 26))
        lines.append(f"   {n:<5} {w*100:>6.1f}%  {bar}")

    lines.append(
        "\n   * Pitch-class su componente armonica (batteria esclusa)."
        "\n     Chroma = nota a meno dell'ottava; "
        "enarmonia dipende dalla tonalità."
    )

    return "\n".join(lines)
