#!/usr/bin/env python3
"""
notes.py
─────────────────────────────────────────────────────────────────────────────
NOTE PIÙ USATE nel brano — terzo ramo parallelo (NON Canary, NON il genere).

PERCHÉ NON "trascrizione nota per nota":
  Su un mix finito (voce+chitarra+basso+batteria+produzione) la trascrizione
  polifonica è un problema di ricerca APERTO: il kick si confonde col basso,
  il riverbero genera note-fantasma, il melisma vocale si spalma. Contare
  quelle "note" = contare rumore.

COSA FACCIAMO (robusto, standard MIR):
  CHROMAGRAM → istogramma delle 12 pitch-class (C, C#, D, ... B).
  Quanta energia tonale, su TUTTO il brano, sta su ciascuna delle 12 note,
  indipendentemente dall'ottava. È esattamente lo strumento usato dai
  sistemi reali di key/harmony detection.

  + Stima TONALITÀ (Krumhansl-Schmuckler): correla il profilo chroma con
    i 24 profili maggiore/minore. Dà senso all'istogramma:
    "le note più usate" → la scala della tonalità del brano.

PIPELINE:

  ┌──────────────────────────────────────────────────────────────────────┐
  │  [C lib: PCM float32 grezzo]                                         │
  │       → mono, resample @ 22050 Hz                                    │
  │       → HPSS: tieni SOLO la componente ARMONICA                      │
  │         (la batteria/percussioni inquinano la chroma — via)          │
  │       → chroma_cqt  → matrice [12, n_frame]                          │
  │       → somma sul tempo → vettore 12-dim → normalizza → %            │
  │       → ordina → note più usate                                      │
  │       → correla con profili K-S → tonalità stimata                   │
  └──────────────────────────────────────────────────────────────────────┘

TEORIA — perché HPSS armonico:
  Harmonic/Percussive Source Separation scompone lo spettrogramma in
  componente "orizzontale" (toni sostenuti = note) e "verticale"
  (transienti = colpi di batteria). L'hip-hop è denso di percussioni:
  senza questo step l'istogramma è dominato da energia a banda larga
  che NON è una nota. Tenere solo l'armonico = chiedere "quali NOTE",
  non "quanta batteria".

LIMITE ONESTO:
  La chroma dà la pitch-CLASS (C vs C#), NON l'ottava (C3 vs C4).
  Per "le note più usate" è l'astrazione GIUSTA: vuoi sapere che la
  tonica è La, non in quale ottava cade ogni occorrenza.
  Nomi con diesis (C#, D#, ...); l'enarmonia (C# = Db) dipende dalla
  tonalità — riportata a parte.

UPGRADE (se un giorno ti serve la MELODIA vocale nota-per-nota):
  Demucs (separa la voce) → CREPE/pYIN (pitch monofonico sulla voce).
  Dipendenze pesanti, fuori scope qui.

DIPENDENZE: librosa, numpy  (pip puro, niente TensorFlow)
─────────────────────────────────────────────────────────────────────────────
"""

import numpy as np


# 0=C ... 11=B  (convenzione librosa: bin 0 = C)
_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Profili Krumhansl-Schmuckler (pesi di percezione tonale, valori pubblicati)
_KS_MAJOR = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
)
_KS_MINOR = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
)


def _best_key(chroma_vec: np.ndarray) -> dict:
    """
    Correla il vettore chroma (12-dim) con tutte le 12 rotazioni dei
    profili maggiore e minore. Ritorna la tonalità a correlazione massima.
    """
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
                best = {"key": _NOTE_NAMES[shift], "mode": mode, "score": c}
    return best


class NoteDetector:
    """
    Uso:
        nd = NoteDetector()
        result = nd.predict(samples, sr, ch)   # buffer C GREZZO

    `samples`/`sr`/`ch` = esattamente i valori di read_audio_via_c().
    """

    def __init__(self, target_sr: int = 22050):
        # Import locale: se non usi le note, non paghi l'import di librosa
        import librosa  # noqa: F401  (verifica presenza all'init)

        self.target_sr = target_sr

    def predict(
        self,
        samples: np.ndarray,
        sample_rate: int,
        channels: int,
        top_k: int = 5,
    ) -> dict:
        """
        Ritorna:
          {
            "key": "F#", "mode": "minor", "key_confidence": 0.78,
            "histogram": [("F#",0.18),("C#",0.15), ... 12 voci ...],
            "top_k":     [("F#",0.18),("C#",0.15),("A",0.12),...],
          }
        """
        import librosa

        # 1. mono
        x = samples.astype(np.float32)
        if channels > 1:
            x = x.reshape(-1, channels).mean(axis=1)

        # 2. resample → target_sr
        if sample_rate != self.target_sr:
            x = librosa.resample(
                x, orig_sr=sample_rate, target_sr=self.target_sr
            )

        # 3. HPSS — tieni SOLO l'armonico (via batteria/percussioni)
        x_harm = librosa.effects.harmonic(x, margin=4.0)

        # 4. chromagram CQT  → [12, n_frame]
        chroma = librosa.feature.chroma_cqt(
            y=x_harm, sr=self.target_sr
        )

        # 5. energia totale per pitch-class → normalizza a somma 1
        energy = chroma.sum(axis=1)
        total = energy.sum()
        if total <= 0:
            return {
                "key": None, "mode": None, "key_confidence": 0.0,
                "histogram": [], "top_k": [],
            }
        weights = energy / total

        hist = sorted(
            ((_NOTE_NAMES[i], round(float(weights[i]), 4)) for i in range(12)),
            key=lambda kv: kv[1],
            reverse=True,
        )

        # 6. tonalità stimata
        key = _best_key(energy)

        return {
            "key": key["key"],
            "mode": key["mode"],
            "key_confidence": round(key["score"], 4),
            "histogram": hist,          # tutte e 12, ordinate
            "top_k": hist[:top_k],
        }


# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT FORMATTATO (stile coerente col tuo print_file_result)
# ─────────────────────────────────────────────────────────────────────────────

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
