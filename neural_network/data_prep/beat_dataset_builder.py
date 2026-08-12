"""
Preprocessing del database PTB-XL (PhysioNet) per creare un dataset di
singoli battiti segmentati, da usare per un modello CNN/GCN che rileva
il posizionamento errato degli elettrodi nell'ECG a 12 derivazioni.

-------
Per ogni registrazione ECG a 12 derivazioni (10 secondi):
  1. Carico il segnale grezzo (wfdb).
  2. Rilevo i picchi R su una derivazione "guida" (default: lead II).
  3. Per ogni picco R trovato, estraggo una finestra sincrona su TUTTE le
     12 derivazioni: [R - 300 ms, R + 600 ms].
  4. Scarto i battiti troppo vicini all'inizio/fine della registrazione
     (finestra che uscirebbe dai 10s).
  5. Salvo ogni battito come file .npy separato (seg000000.npy,
     seg000001.npy, ...) in <output_dir>/segments/, shape (window_len, 12).
     Un file CSV di indice collega ogni file ai metadati (ecg_id, patient_id,
     strat_fold, posizione del picco R, ecc.).
----------------------------------------------------------------
- Il rilevamento dei picchi R viene fatto su UNA sola derivazione guida
  e la stessa posizione temporale viene poi applicata a TUTTE le 12
  derivazioni. Questo è voluto, dato che, se rilevassi i picchi
  indipendentemente su ogni derivazione, uno scambio di elettrodi
  potrebbe disallineare le finestre tra derivazioni, "nascondendo" col
  preprocessing proprio l'anomalia che vogliamo far riconoscere al modello.
- Lo split train/val/test va fatto per PAZIENTE (patient_id), non per
  battito né per registrazione, altrimenti avremmo data leakage. PTB-XL
  fornisce già 'strat_fold' pensato per questo: fold 1-8 = train,
  9 = val, 10 = test (convenzione standard del paper originale di PTB-XL).
"""

import argparse
import os

from preprocessing import preprocess_ecg
from pan_tompkins_algo import detect_r_peaks_pan_tompkins

import numpy as np
import pandas as pd
import wfdb
from tqdm import tqdm

LEAD_NAMES = ["I", "II", "III", "AVR", "AVL", "AVF",
              "V1", "V2", "V3", "V4", "V5", "V6"]

# Numero di cifre usate nel nome dei file seg000000.npy -> fino a 10^7 - 1 battiti
FILENAME_DIGITS = 7


# --------------------------------------------------------------------------
# 1. Caricamento metadati e segnali PTB-XL
# --------------------------------------------------------------------------


def load_ptbxl_metadata(ptbxl_root: str) -> pd.DataFrame:

    path = os.path.join(ptbxl_root, "ptbxl_database.csv")
    # legge il CSV in un DataFrame e usa 'ecg_id' come indice
    df = pd.read_csv(path, index_col="ecg_id")
    return df


def load_raw_signal(ptbxl_root: str, row: pd.Series, sampling_rate: int) -> np.ndarray:
    """
    Carica il segnale grezzo per una singola registrazione.
    Ritorna un array shape (n_samples, 12).
    """
    # seleziona il percorso relativo in base al sampling rate richiesto
    if sampling_rate == 100:
        rel_path = row["filename_lr"]
    elif sampling_rate == 500:
        rel_path = row["filename_hr"]
    else:
        raise ValueError("sampling_rate deve essere 100 o 500 (come da file PTB-XL)")

    # costruisce il percorso assoluto al record WFDB
    record_path = os.path.join(ptbxl_root, rel_path)
    signal, meta = wfdb.rdsamp(record_path)
    # verifica che l'ordine delle derivazioni corrisponda a quello atteso
    assert meta["sig_name"] == LEAD_NAMES, (
        f"Ordine derivazioni sbagliato: {meta['sig_name']}"
    )
    return signal.astype(np.float32)


# --------------------------------------------------------------------------
# 2. Rilevamento picchi R
# --------------------------------------------------------------------------


def detect_r_peaks(signal_12lead: np.ndarray, fs: int, guide_lead: str = "II"):
    """
    Rileva i picchi R sulla derivazione guida.
    Applica preprocessing prima del rilevamento.
    Ritorna un array di indici campione (int). Vuoto se il rilevamento fallisce.
    """
    # ottiene l'indice della derivazione guida nell'array 12-lead
    lead_idx = LEAD_NAMES.index(guide_lead)
    # estrae la singola derivazione da analizzare
    lead_signal = signal_12lead[:, lead_idx]
    # applica preprocessing al segnale prima del rilevamento dei picchi
    lead_signal = preprocess_ecg(lead_signal, fs)

    try:
        # usa l'algoritmo Pan-Tompkins per rilevare i picchi R
        r_peaks = detect_r_peaks_pan_tompkins(lead_signal, fs)
    except Exception:
        r_peaks = np.array([], dtype=int)

    return r_peaks


# --------------------------------------------------------------------------
# 3. Segmentazione dei battiti
# --------------------------------------------------------------------------


def segment_beats(signal_12lead: np.ndarray, r_peaks: np.ndarray, fs: int,
                   pre_ms: float = 400.0, post_ms: float = 600.0):
    """
    Estrae, per ogni picco R, una finestra [R-pre_ms, R+post_ms] su tutte
    le 12 derivazioni.

    Ritorna:
        beats: array shape (n_beats_validi, window_len, 12)
        used_peaks: indici campione dei picchi R effettivamente usati
    """
    n_samples = signal_12lead.shape[0]
    # calcolo dei campioni corrispondenti a pre_ms e post_ms
    pre_samples = int(round(pre_ms / 1000.0 * fs))
    post_samples = int(round(post_ms / 1000.0 * fs))
    # lunghezza della finestra in campioni
    window_len = pre_samples + post_samples

    beats = []
    used_peaks = []
    for r in r_peaks:
        start = r - pre_samples
        end = r + post_samples
        # scarta finestre che escono dai bordi del segnale
        if start < 0 or end > n_samples:
            continue
        # aggiunge alla lista la finestra (tutte le 12 derivazioni)
        beats.append(signal_12lead[start:end, :])
        used_peaks.append(r)

    # se non ci sono battiti validi ritorna array vuoto con shape corretta
    if len(beats) == 0:
        return (np.empty((0, window_len, 12), dtype=np.float32),
                np.array([], dtype=int))

    # impacchetta le finestre in un array numpy e ritorna anche i picchi usati
    return np.stack(beats).astype(np.float32), np.array(used_peaks, dtype=int)


# --------------------------------------------------------------------------
# 4. Pipeline completa: costruzione del dataset
# --------------------------------------------------------------------------


def build_dataset(ptbxl_root: str, output_dir: str, sampling_rate: int = 500,
                   guide_lead: str = "II", pre_ms: float = 400.0,
                   post_ms: float = 600.0, min_beats_per_record: int = 1,
                   limit: int = None):
    """
    Processa l'intero database PTB-XL e salva:
      - <output_dir>/segments/seg0000000.npy, seg0000001.npy, ...
      - <output_dir>/beats_index.csv    : indice con metadati per ogni battito
      - <output_dir>/skipped_records.csv: registrazioni escluse e motivo
    """
    segments_dir = os.path.join(output_dir, "segments")
    os.makedirs(segments_dir, exist_ok=True)

    meta_df = load_ptbxl_metadata(ptbxl_root)
    # se è stato passato un limite, prende solo le prime N righe
    if limit is not None:
        meta_df = meta_df.iloc[:limit]

    fs = sampling_rate
    index_rows = []
    skipped_records = []
    global_beat_counter = 0

    # itera su tutte le registrazioni nel DataFrame
    for ecg_id, row in tqdm(meta_df.iterrows(), total=len(meta_df),
                             desc="Processing registrazioni"):
        try:
            # carica il segnale grezzo per la registrazione corrente
            signal = load_raw_signal(ptbxl_root, row, sampling_rate)
        except Exception as e:
            # registra l'errore e salta la registrazione
            skipped_records.append((ecg_id, f"errore_lettura: {e}"))
            continue

        # rileva i picchi R sulla derivazione guida
        r_peaks = detect_r_peaks(signal, fs, guide_lead=guide_lead)
        # se ci sono troppo pochi picchi considera la registrazione non valida
        if len(r_peaks) < min_beats_per_record:
            skipped_records.append((ecg_id, "picchi_R_insufficienti"))
            continue

        # segmenta i battiti usando i picchi R trovati
        beats, used_peaks = segment_beats(signal, r_peaks, fs, pre_ms, post_ms)
        # se la segmentazione non ha prodotto battiti validi, salta
        if beats.shape[0] == 0:
            skipped_records.append((ecg_id, "nessun_battito_valido_dopo_bordo"))
            continue

        # salva ciascun battito come file .npy e aggiunge una riga all'indice
        for local_idx, r_sample in enumerate(used_peaks):
            # genera il nome file basato sul contatore globale
            filename = f"seg{global_beat_counter:0{FILENAME_DIGITS}d}.npy"
            # percorso completo del file
            filepath = os.path.join(segments_dir, filename)
            # salva il singolo battito
            np.save(filepath, beats[local_idx])

            # aggiunge i metadati corrispondenti alla lista di righe
            index_rows.append({
                "filename": filename,
                "ecg_id": ecg_id,
                "patient_id": row["patient_id"],
                "beat_idx_in_record": local_idx,
                "r_peak_sample": int(r_sample),
                "strat_fold": row["strat_fold"],
                "guide_lead": guide_lead,
                "sampling_rate": sampling_rate,
            })
            # incrementa il contatore globale dopo aver salvato il file
            global_beat_counter += 1

    # costruisce il DataFrame indice dai dizionari raccolti
    index_df = pd.DataFrame(index_rows)
    # percorso CSV di output per l'indice dei battiti
    index_csv_path = os.path.join(output_dir, "beats_index.csv")
    # salva il CSV senza colonna indice
    index_df.to_csv(index_csv_path, index=False)

    # costruisce e salva il CSV delle registrazioni saltate con motivo
    skipped_df = pd.DataFrame(skipped_records, columns=["ecg_id", "motivo"])
    skipped_csv_path = os.path.join(output_dir, "skipped_records.csv")
    skipped_df.to_csv(skipped_csv_path, index=False)

    # messaggi di riepilogo
    print(f"\nCompletato.")
    print(f"  Battiti totali salvati : {global_beat_counter}")
    print(f"  Registrazioni saltate  : {len(skipped_records)}")
    print(f"  Cartella segmenti      : {segments_dir}")
    print(f"  Indice battiti (CSV)   : {index_csv_path}")
    print(f"  Registrazioni saltate  : {skipped_csv_path}")

    # ritorna i percorsi di interesse per eventuali chiamanti
    return segments_dir, index_csv_path


# --------------------------------------------------------------------------
# 5. CLI
# --------------------------------------------------------------------------


def main():
    # crea il parser degli argomenti usando il docstring come descrizione
    parser = argparse.ArgumentParser(description=__doc__)
    # argomento: cartella radice PTB-XL
    parser.add_argument("--ptbxl_root", required=True,
                         help="Cartella radice del database PTB-XL locale "
                              "(quella che contiene ptbxl_database.csv)")
    # argomento: cartella di output per segments/ e beats_index.csv
    parser.add_argument("--output_dir", required=True,
                         help="Cartella dove salvare segments/ e beats_index.csv")
    # argomento: sampling rate, 100 o 500 Hz
    parser.add_argument("--sampling_rate", type=int, default=500, choices=[100, 500],
                         help="100 Hz o 500 Hz (come da file PTB-XL originali)")
    # argomento: derivazione guida per il rilevamento dei picchi R
    parser.add_argument("--guide_lead", type=str, default="II",
                         choices=LEAD_NAMES,
                         help="Derivazione usata per il rilevamento dei picchi R")
    # argomenti: dimensione finestra pre e post in ms
    parser.add_argument("--pre_ms", type=float, default=300.0)
    parser.add_argument("--post_ms", type=float, default=600.0)
    # argomento: numero minimo di picchi R richiesti per considerare valida una registrazione
    parser.add_argument("--min_beats_per_record", type=int, default=1)
    # argomento: limite per debugging (processa solo le prime N registrazioni)
    parser.add_argument("--limit", type=int, default=None,
                         help="Solo per debug: processa solo le prime N registrazioni")
    # effettua il parsing degli argomenti dalla CLI
    args = parser.parse_args()

    # invoca la pipeline principale con i parametri parsati
    build_dataset(
        ptbxl_root=args.ptbxl_root,
        output_dir=args.output_dir,
        sampling_rate=args.sampling_rate,
        guide_lead=args.guide_lead,
        pre_ms=args.pre_ms,
        post_ms=args.post_ms,
        min_beats_per_record=args.min_beats_per_record,
        limit=args.limit,
    )


if __name__ == "__main__":
    # esegue main quando il file è eseguito come script
    main()
