"""
Trasformazioni per simulare, a partire da un ECG a 12 derivazioni
registrato correttamente, l'effetto di uno scambio di elettrodi
sul tracciato.

BASE TEORICA
------------
Le 6 derivazioni degli arti sono definite a partire dai potenziali
"grezzi" dei tre elettrodi (RA=braccio destro, LA=braccio sinistro,
LL=gamba sinistra) tramite le equazioni di Einthoven e Goldberger:

    I   = LA - RA
    II  = LL - RA
    III = LL - LA
    aVR = RA - (LA + LL) / 2 = -(I + II) / 2
    aVL = LA - (RA + LL) / 2 = (I - III) / 2
    aVF = LL - (RA + LA) / 2 = (II + III) / 2

Le derivazioni precordiali V1-V6 sono invece riferite al terminale
centrale di Wilson: WCT = (RA + LA + LL) / 3.

Se due cavi degli arti vengono scambiati fisicamente, verranno modificate
tutte le derivazione in cui sono coinvolti i due elettrodi scambiati.
Sostituendo questo nelle equazioni sopra si ottiene una combinazione
lineare (permutazione + cambi di segno) delle derivazioni originali,
verificata numericamente (vedi test in fondo al file). Poiche' WCT e'
una somma simmetrica di RA+LA+LL, e' invariante rispetto a QUALSIASI
scambio tra i tre elettrodi degli arti: V1-V6 non cambiano mai in questi casi.

Lo scambio di due cavi precordiali (es. V1-V2) e' invece un caso
diverso e più semplice: non tocca WCT (che dipende solo da RA/LA/LL),
quindi equivale a un semplice scambio di canale tra i due V coinvolti,
con tutto il resto invariato.

TRASFORMAZIONI IMPLEMENTATE
----------------------------
- "RA_LA": scambio elettrodi braccio destro / braccio sinistro
    I' = -I,  II' = III,  III' = II,  aVR' = aVL,  aVL' = aVR,  aVF' = aVF
- "LA_LL": scambio elettrodi braccio sinistro / gamba sinistra
    I' = II,  II' = I,  III' = -III,  aVR' = aVR,  aVL' = aVF,  aVF' = aVL
- "RA_LL": scambio elettrodi braccio destro / gamba sinistra
    I' = -III, II' = -II, III' = -I, aVR' = aVF, aVL' = aVL, aVF' = aVR
- "V1_V2", "V2_V3", "V3_V4", "V4_V5", "V5_V6": scambio tra due cavi
    precordiali adiacenti (il caso più comune in pratica clinica)

NOTA: lo scambio che coinvolge l'elettrodo di riferimento/massa (RL,
right leg) con LL, non e' incluso in quanto non comporta una modifica
dei tracciati.
"""

import numpy as np

LEAD_NAMES = ["I", "II", "III", "AVR", "AVL", "AVF",
              "V1", "V2", "V3", "V4", "V5", "V6"]

_IDX = {name: i for i, name in enumerate(LEAD_NAMES)}


def _limb_swap_RA_LA(sig: np.ndarray) -> np.ndarray:
    out = sig.copy()
    I, II, III = sig[..., _IDX["I"]], sig[..., _IDX["II"]], sig[..., _IDX["III"]]
    aVR, aVL = sig[..., _IDX["AVR"]], sig[..., _IDX["AVL"]]
    out[..., _IDX["I"]] = -I
    out[..., _IDX["II"]] = III
    out[..., _IDX["III"]] = II
    out[..., _IDX["AVR"]] = aVL
    out[..., _IDX["AVL"]] = aVR
    # AVF e V1-V6 invariati
    return out


def _limb_swap_LA_LL(sig: np.ndarray) -> np.ndarray:
    out = sig.copy()
    I, II, III = sig[..., _IDX["I"]], sig[..., _IDX["II"]], sig[..., _IDX["III"]]
    aVL, aVF = sig[..., _IDX["AVL"]], sig[..., _IDX["AVF"]]
    out[..., _IDX["I"]] = II
    out[..., _IDX["II"]] = I
    out[..., _IDX["III"]] = -III
    out[..., _IDX["AVL"]] = aVF
    out[..., _IDX["AVF"]] = aVL
    # AVR e V1-V6 invariati
    return out


def _limb_swap_RA_LL(sig: np.ndarray) -> np.ndarray:
    out = sig.copy()
    I, II, III = sig[..., _IDX["I"]], sig[..., _IDX["II"]], sig[..., _IDX["III"]]
    aVR, aVF = sig[..., _IDX["AVR"]], sig[..., _IDX["AVF"]]
    out[..., _IDX["I"]] = -III
    out[..., _IDX["II"]] = -II
    out[..., _IDX["III"]] = -I
    out[..., _IDX["AVR"]] = aVF
    out[..., _IDX["AVF"]] = aVR
    # AVL e V1-V6 invariati
    return out


def _make_precordial_swap(lead_a: str, lead_b: str):
    def _swap(sig: np.ndarray) -> np.ndarray:
        out = sig.copy()
        ia, ib = _IDX[lead_a], _IDX[lead_b]
        out[..., ia] = sig[..., ib]
        out[..., ib] = sig[..., ia]
        return out
    return _swap


def _identity(sig: np.ndarray) -> np.ndarray:
    return sig.copy()


# Registro di tutte le trasformazioni disponibili, per nome.
MISPLACEMENT_TRANSFORMS = {
    "normal": _identity,
    "RA_LA": _limb_swap_RA_LA,
    "LA_LL": _limb_swap_LA_LL,
    "RA_LL": _limb_swap_RA_LL,
    "V1_V2": _make_precordial_swap("V1", "V2"),
    "V2_V3": _make_precordial_swap("V2", "V3"),
    "V3_V4": _make_precordial_swap("V3", "V4"),
    "V4_V5": _make_precordial_swap("V4", "V5"),
    "V5_V6": _make_precordial_swap("V5", "V6"),
}


def apply_transform(sig: np.ndarray, transform_name: str) -> np.ndarray:
    """
    Applica la trasformazione richiesta a un array le cui ultime
    dimensioni sono le 12 derivazioni nell'ordine LEAD_NAMES.
    Funziona sia su un singolo battito (window_len, 12) sia su
    un'intera registrazione (n_samples, 12).
    """
    if transform_name not in MISPLACEMENT_TRANSFORMS:
        raise ValueError(
            f"Trasformazione '{transform_name}' sconosciuta. "
            f"Disponibili: {list(MISPLACEMENT_TRANSFORMS.keys())}"
        )
    assert sig.shape[-1] == 12, "L'ultima dimensione deve avere 12 derivazioni"
    return MISPLACEMENT_TRANSFORMS[transform_name](sig)
