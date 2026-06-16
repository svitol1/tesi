import importlib
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import wfdb
import matplotlib.pyplot as plt

try:
    from ...files.preprocessing import preprocess_ecg
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    preprocess_ecg = importlib.import_module("files.preprocessing").preprocess_ecg


def load_ecg_segment(record_name: str = "101", pn_dir: str = "mitdb", lead: int = 0):
    record = wfdb.rdrecord(f"../../physionet.org/files/{pn_dir}/1.0.0/{record_name}")
    signal = np.asarray(record.p_signal[:, lead], dtype=float)
    fs = float(record.fs)
    signal = preprocess_ecg(signal, fs)
    return signal, fs


def _pad_segment(signal: np.ndarray, target_length: int) -> np.ndarray:
    if len(signal) >= target_length:
        return signal[:target_length]
    padded = np.zeros(target_length, dtype=float)
    padded[: len(signal)] = signal
    return padded


def create_dataset(
    output_dir: str = "dataset",
    pn_dir: str = "mitdb",
    lead: int = 0,
    record_names: list[str] | None = None,
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if record_names is None:
        record_names = wfdb.get_record_list(pn_dir)

    # Dizionario di mappatura standard AAMI per 4 classi
    label_map = {
        "N": 0,
        "L": 0,
        "R": 0,  # Normali / Blocchi di branca
        "A": 1,
        "a": 1,
        "J": 1,
        "S": 1,  # Sopraventricolari
        "V": 2,
        "E": 2,  # Ventricolari
        "F": 3,  # Fusione
    }

    # Lista in cui raccoglieremo i metadati [percorso_file, etichetta]
    dataset_records = []

    samples_per_segment = 2500

    for record_name in record_names:
        print(f"Processing record: {record_name}")

        # Carica il segnale
        signal, fs = load_ecg_segment(
            record_name=record_name,
            pn_dir=pn_dir,
            lead=lead
        )

        # Carica le annotazioni del record corrente
        try:
            ann = wfdb.rdann(f"../../physionet.org/files/{pn_dir}/1.0.0/{record_name}", "atr")
            beat_samples = ann.sample
            beat_symbols = ann.symbol
        except Exception as e:
            print(f"Impossibile leggere le annotazioni per {record_name}: {e}. Salto il record.")
            continue

        record_dir = output_path / record_name
        record_dir.mkdir(parents=True, exist_ok=True)

        segment_count = int(np.ceil(len(signal) / samples_per_segment))

        for index in range(segment_count):
            start = index * samples_per_segment
            end = start + samples_per_segment

            # Trova quali battiti cadono all'interno di questo specifico segmento
            idx_in_segment = np.where((beat_samples >= start) & (beat_samples < end))[0]
            symbols_in_segment = [beat_symbols[i] for i in idx_in_segment]

            # Mappa i simboli medici nelle 4 classi numeriche (0-3)
            classes_in_segment = [label_map[s] for s in symbols_in_segment if s in label_map]

            # Se nel segmento non ci sono battiti rilevanti o conosciuti, saltiamo il segmento
            if len(classes_in_segment) == 0:
                continue

            # Applichiamo la regola della priorità medica: se c'è una qualsiasi anomalia,
            # il segmento prende l'etichetta del battito più "grave" (valore massimo tra 0-3)
            segment_label = max(classes_in_segment)

            # Salva il file del segnale
            segment = _pad_segment(signal[start:end], samples_per_segment)
            file_name = f"seg{index:03d}.npy"
            relative_file_path = f"{record_name}/{file_name}"

            np.save(record_dir / file_name, segment)

            # Salva le informazioni nella nostra lista dei metadati
            dataset_records.append(
                {"file_path": relative_file_path, "label": segment_label}
            )

    # 3. Creazione del file CSV unico dentro la cartella del dataset
    df = pd.DataFrame(dataset_records)
    df.to_csv(output_path / "metadata.csv", index=False)
    print(f"Dataset creato con successo! Generati {len(df)} segmenti validi.")

    return output_path


if __name__ == "__main__":
    #create_dataset()
    # Per testare la visualizzazione di un segmento
    segment = np.load("dataset/208/seg145.npy")
    plt.plot(segment)
    plt.title("Segmento di ECG")
    plt.xlabel("Campioni")
    plt.ylabel("Amplitude")
    plt.show()