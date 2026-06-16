import importlib
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import wfdb
import matplotlib.pyplot as plt

try:
    from preprocessing import preprocess_ecg
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    preprocess_ecg = importlib.import_module("files.preprocessing").preprocess_ecg


def load_ecg_segment(record_name: str = "101", pn_dir: str = "mitdb", lead: int = 1):
    record = wfdb.rdrecord(f"../../../dajeee/tirocinio/physionet.org/files/{pn_dir}/1.0.0/{record_name}")
    signal = np.asarray(record.p_signal[:, lead], dtype=float)
    fs = float(record.fs)
    signal = preprocess_ecg(signal, fs)
    return signal, fs


def load_ecg_multilead(record_name: str = "101", pn_dir: str = "mitdb", leads: tuple[int, int] = (0, 1)):
    signals = []
    fs = None
    for lead in leads:
        signal, fs = load_ecg_segment(record_name=record_name, pn_dir=pn_dir, lead=lead)
        signals.append(signal)
    return np.stack(signals, axis=0), fs


def _pad_segment(signal: np.ndarray, target_length: int) -> np.ndarray:
    if len(signal) >= target_length:
        return signal[:target_length]
    padded = np.zeros(target_length, dtype=float)
    padded[: len(signal)] = signal
    return padded


def create_dataset(
    output_dir: str = "dataset",
    pn_dir: str = "mitdb",
    lead: int = 1,
    record_names: list[str] | None = None,
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if record_names is None:
        record_names = wfdb.get_record_list(pn_dir)
        # togliamo l'ultimo record per testare la visualizzazione
        record_names.pop()

    # Dizionario di mappatura standard AAMI per 4 classi
    label_map = {
        "N": 0,
        "L": 0,
        "R": 0,  # Normali / Blocchi di branca N
        "A": 1,
        "a": 1,
        "J": 1,
        "S": 1,  # Sopraventricolari SVEB
        "V": 2,
        "E": 2,  # Ventricolari VEB
        "F": 3,  # Fusione F
    }

    # Lista in cui raccoglieremo i metadati [percorso_file, etichetta]
    dataset_records = []

    # Manteniamo una lunghezza fissa per poter addestrare la CNN,
    # ma ogni esempio è centrato su un singolo battito annotato.
    # Finestra di circa 1 s a 360 Hz, con centro esatto sul picco R.
    samples_per_segment = 360

    for record_name in record_names:
        print(f"Processing record: {record_name}")

        # Carica entrambe le derivazioni e mantienile allineate sullo stesso asse temporale.
        signal, fs = load_ecg_multilead(
            record_name=record_name,
            pn_dir=pn_dir,
            leads=(0, 1),
        )

        # Carica le annotazioni del record corrente
        try:
            ann = wfdb.rdann(f"../../../dajeee/tirocinio/physionet.org/files/{pn_dir}/1.0.0/{record_name}", "atr")
            r_peaks = ann.sample
            beat_symbols = ann.symbol
        except Exception as e:
            print(f"Impossibile leggere le annotazioni per {record_name}: {e}. Salto il record.")
            continue

        record_dir = output_path / record_name
        record_dir.mkdir(parents=True, exist_ok=True)

        # Creiamo un campione per ogni battito annotato, centrando il segmento su quel battito.
        # In questo modo l'etichetta del segmento coincide con l'annotazione del battito stesso.
        half_window = samples_per_segment // 2

        for beat_index, (peak, beat_symbol) in enumerate(zip(r_peaks, beat_symbols)):
            if beat_symbol not in label_map:
                continue

            segment_label = label_map[beat_symbol]

            center = int(peak)
            start = center - half_window
            end = start + samples_per_segment

            # Salva il file del segnale centrato sul battito annotato
            segment = np.stack(
                [_pad_segment(channel[start:end], samples_per_segment) for channel in signal],
                axis=0,
            )
            file_name = f"seg{beat_index:05d}.npy"
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
    segment = np.load("dataset/200/seg00409.npy")
    if segment.ndim == 2 and segment.shape[0] == 2:
        fig, axes = plt.subplots(2, 1, sharex=True, figsize=(12, 6))
        lead_names = ["Derivazione 1", "Derivazione 2"]

        for ax, lead_signal, name in zip(axes, segment, lead_names):
            ax.plot(lead_signal)
            ax.set_title(name)
            ax.set_ylabel("Amplitude")

        axes[-1].set_xlabel("Campioni")
        fig.suptitle("Segmento di ECG")
        plt.tight_layout()
        plt.show()
    else:
        plt.plot(segment)
        plt.title("Segmento di ECG")
        plt.xlabel("Campioni")
        plt.ylabel("Amplitude")
        plt.show()