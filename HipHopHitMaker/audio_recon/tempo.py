#!/usr/bin/env python3
"""
tempo.py
─────────────────────────────────────────────────────────────────────────────
BPM e TEMPO del brano — quinto ramo parallelo (NON Canary).

DESIGN SIMMETRICO:
  notes.py usa la componente ARMONICA (le note vivono lì).
  tempo.py usa la componente PERCUSSIVA (i beat vivono lì).
  HPSS isola i transienti ritmici → beat tracking più pulito su mix
  carichi di voce/synth.

COSA RITORNA, su 3 livelli:
  1. BPM           → battiti al minuto (numero). Problema risolto.
  2. Indicazione   → Largo/Adagio/Andante/Moderato/Allegro/Presto dal BPM.
  3. Metro (4/4?)  → EURISTICA bassa-confidenza. librosa NON rileva il
                     time signature in modo affidabile: problema aperto.
                     Riportato come ipotesi, MAI come verità.

TEORIA — errore di ottava (la trappola classica):
  Ogni beat tracker confonde tempo a fattori 2 e 3: riporta 72 invece di
  144, o 174 invece di 87. È IL fallimento documentato della tempo
  estimation. NON lo correggiamo in silenzio: esponiamo il BPM primario
  E i candidati (metà/doppio/×1.5) così la decisione resta informata.

TEORIA — stabilità del tempo:
  librosa.feature.tempo(aggregate=None) dà il tempo frame-per-frame.
  Deviazione standard bassa → beat programmato (electronic/hip-hop).
  Alta → esecuzione dal vivo / rubato. Segnale utile, quasi gratis.

PIPELINE:
  ┌──────────────────────────────────────────────────────────────────────┐
  │  [C lib: PCM grezzo] → mono → resample 22050                         │
  │       → HPSS: componente PERCUSSIVA                                   │
  │       → onset envelope                                                │
  │       → beat_track  → BPM globale + griglia beat                      │
  │       → tempo(aggregate=None) → BPM nel tempo → std → stabilità       │
  │       → candidati ottava (×0.5 ×2 ×1.5 ×0.667)                        │
  │       → autocorrelazione battiti → ipotesi metro 3 vs 4 (debole)      │
  └──────────────────────────────────────────────────────────────────────┘

LIMITE ONESTO:
  Il metro è un'EURISTICA. Pop/hip-hop è quasi sempre 4/4: l'ipotesi
  sarà spesso giusta per caso, non per analisi robusta. Fidati del BPM,
  tratta il metro come spunto.

DIPENDENZE: librosa, numpy  (già installate per notes.py)
─────────────────────────────────────────────────────────────────────────────
"""

import numpy as np


# Range convenzionali (variano per fonte: indicativi, non assoluti)
_TEMPO_MARKS = [
    (0,    66,  "Largo"),
    (66,   76,  "Adagio"),
    (76,   108, "Andante"),
    (108,  120, "Moderato"),
    (120,  168, "Allegro"),
    (168,  200, "Presto"),
    (200,  1e9, "Prestissimo"),
]


def _mark(bpm: float) -> str:
    for lo, hi, name in _TEMPO_MARKS:
        if lo <= bpm < hi:
            return name
    return "?"


class TempoDetector:
    """
    Uso:
        td = TempoDetector()
        result = td.predict(samples, sr, ch)   # buffer C GREZZO
    """

    def __init__(self, target_sr: int = 22050):
        import librosa  # noqa: F401  (verifica presenza all'init)
        self.target_sr = target_sr

    def predict(
        self,
        samples: np.ndarray,
        sample_rate: int,
        channels: int,
    ) -> dict:
        """
        Ritorna:
          {
            "bpm": 144.0,
            "mark": "Allegro",
            "stability": "steady",          # steady | variable
            "bpm_std": 2.3,
            "octave_candidates": [72.0, 96.0, 216.0, 288.0],
            "meter_guess": "4/4",           # EURISTICA bassa confidenza
            "meter_confidence": "bassa",
            "n_beats": 213,
          }
        """
        import librosa

        # 1. mono + resample
        x = samples.astype(np.float32)
        if channels > 1:
            x = x.reshape(-1, channels).mean(axis=1)
        if sample_rate != self.target_sr:
            x = librosa.resample(
                x, orig_sr=sample_rate, target_sr=self.target_sr
            )

        sr = self.target_sr

        # 2. HPSS — componente PERCUSSIVA (i beat vivono lì)
        x_perc = librosa.effects.percussive(x, margin=3.0)

        # 3. onset envelope sul percussivo
        onset_env = librosa.onset.onset_strength(y=x_perc, sr=sr)

        # 4. beat tracking → BPM globale + griglia
        tempo_arr, beats = librosa.beat.beat_track(
            onset_envelope=onset_env, sr=sr
        )
        bpm = float(np.atleast_1d(tempo_arr)[0])

        # 5. stabilità: BPM su FINESTRE da ~12s.
        #    Ogni finestra può cadere su un'ottava diversa (80/160/240):
        #    la ripiego sull'ottava del BPM globale PRIMA della std,
        #    altrimenti misuro i salti di ottava, non il drift reale.
        win = 12 * sr
        _FOLD = (1.0, 0.5, 2.0, 1.5, 2.0 / 3.0, 1.0 / 3.0, 3.0)
        folded = []
        for s in range(0, max(len(x_perc) - win, 1), win):
            seg = x_perc[s:s + win]
            if len(seg) < sr * 4:          # scarta code < 4s
                continue
            oe = librosa.onset.onset_strength(y=seg, sr=sr)
            w = float(np.atleast_1d(
                librosa.feature.tempo(onset_envelope=oe, sr=sr)
            )[0])
            if w <= 0:
                continue
            # porta w il più vicino possibile a bpm globale
            best = min(_FOLD, key=lambda f: abs(w * f - bpm))
            folded.append(w * best)
        bpm_std = float(np.std(folded)) if len(folded) >= 2 else 0.0
        # soglia pragmatica: drift <3 BPM (post-fold) = beat programmato
        stability = "steady" if bpm_std < 3.0 else "variable"

        # 6. candidati errore-di-ottava (NON corretti: esposti)
        cands = sorted({
            round(bpm * f, 1)
            for f in (0.5, 2.0, 1.5, 2.0 / 3.0)
            if 30.0 <= bpm * f <= 320.0
        })

        # 7. ipotesi metro 3 vs 4 — autocorrelazione dell'onset env
        #    sul periodo del beat. EURISTICA: bassa confidenza.
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
        """
        Confronta l'energia media degli onset raggruppati a battute di 4
        vs battute di 3. Il primo movimento (downbeat) è tipicamente più
        forte: il raggruppamento con il contrasto downbeat/resto maggiore
        vince. DEBOLE per costruzione — solo uno spunto.
        """
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


# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT FORMATTATO (stile coerente col tuo print_file_result)
# ─────────────────────────────────────────────────────────────────────────────

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
