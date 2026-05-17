#!/usr/bin/env python3

import numpy as np


_TEMPO_MARKS = [
    (0, 66, "Largo"),
    (66, 76, "Adagio"),
    (76, 108, "Andante"),
    (108, 120, "Moderato"),
    (120, 168, "Allegro"),
    (168, 200, "Presto"),
    (200, 1e9, "Prestissimo"),
]


def _mark(bpm: float) -> str:
    for lo, hi, name in _TEMPO_MARKS:
        if lo <= bpm < hi:
            return name

    return "?"


class TempoDetector:
    def __init__(self, target_sr: int = 22050):
        import librosa

        self.target_sr = target_sr

    def predict(
        self,
        samples: np.ndarray,
        sample_rate: int,
        channels: int,
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

        sr = self.target_sr

        x_perc = librosa.effects.percussive(x, margin=3.0)

        onset_env = librosa.onset.onset_strength(y=x_perc, sr=sr)

        tempo_arr, beats = librosa.beat.beat_track(
            onset_envelope=onset_env,
            sr=sr
        )

        bpm = float(np.atleast_1d(tempo_arr)[0])

        win = 12 * sr
        _FOLD = (1.0, 0.5, 2.0, 1.5, 2.0 / 3.0, 1.0 / 3.0, 3.0)
        folded = []

        for s in range(0, max(len(x_perc) - win, 1), win):
            seg = x_perc[s:s + win]

            if len(seg) < sr * 4:
                continue

            oe = librosa.onset.onset_strength(y=seg, sr=sr)

            w = float(np.atleast_1d(
                librosa.feature.tempo(onset_envelope=oe, sr=sr)
            )[0])

            if w <= 0:
                continue

            best = min(_FOLD, key=lambda f: abs(w * f - bpm))
            folded.append(w * best)

        bpm_std = float(np.std(folded)) if len(folded) >= 2 else 0.0
        stability = "steady" if bpm_std < 3.0 else "variable"

        cands = sorted({
            round(bpm * f, 1)
            for f in (0.5, 2.0, 1.5, 2.0 / 3.0)
            if 30.0 <= bpm * f <= 320.0
        })

        meter = self._meter_heuristic(onset_env, beats)

        return {
            "bpm": round(bpm, 1),
            "mark": _mark(bpm),
            "stability": stability,
            "bpm_std": round(bpm_std, 2),
            "octave_candidates": cands,
            "meter_guess": meter,
            "meter_confidence": "bassa",
            "n_beats": int(len(np.atleast_1d(beats))),
        }

    @staticmethod
    def _meter_heuristic(onset_env: np.ndarray, beats: np.ndarray) -> str:
        beats = np.atleast_1d(beats)

        if len(beats) < 8:
            return "?"

        strengths = onset_env[beats]
        best, choice = -1.0, "?"

        for m in (4, 3):
            n = (len(strengths) // m) * m

            if n < m:
                continue

            grid = strengths[:n].reshape(-1, m)
            downbeat = grid[:, 0].mean()
            rest = grid[:, 1:].mean()
            contrast = (downbeat - rest) / (downbeat + rest + 1e-9)

            if contrast > best:
                best, choice = contrast, f"{m}/4"

        return choice


def format_tempo_block(result: dict) -> str:
    if not result or not result.get("bpm"):
        return ""

    cand = ", ".join(f"{c:g}" for c in result["octave_candidates"])

    lines = [
        "",
        f" TEMPO:  {result['bpm']:g} BPM  ({result['mark']})  "
        f"— {result['n_beats']} beat",
        f"   Stabilità:  {result['stability']}  "
        f"(oscillazione ±{result['bpm_std']:g} BPM)",
        f"   Metro:      {result['meter_guess']}  "
        f"(confidenza {result['meter_confidence']} — euristica)",
        f"   Ottava?:    candidati alternativi → {cand}",
        "\n   * BPM = beat tracking su componente percussiva."
        "\n     I candidati ottava NON sono errori corretti: sono le"
        "\n     letture metà/doppio. Se il brano 'suona' più veloce/lento"
        "\n     del numero, la verità è uno dei candidati.",
    ]

    return "\n".join(lines)
