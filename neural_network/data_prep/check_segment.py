"""Visualizza un segmento normale e uno con malposizionamento.

Il dataset finale viene costruito da ``misplacement_dataset_builder.py``:
ogni registrazione riceve una sola classe e il segmento salvato su disco è
già trasformato in base a quella classe.

Questo script carica due esempi dall'indice finale e li mostra uno sopra
l'altro, così puoi confrontare direttamente un battito normale con uno
malposizionato.
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lead_misplacement import LEAD_NAMES


def load_segment(segments_dir: str, filename: str) -> np.ndarray:
	return np.load(os.path.join(segments_dir, filename))


def plot_segment(ax, segment: np.ndarray, title: str):
	time = np.arange(segment.shape[0])
	offset = 0.0
	step = 3.0 * np.std(segment)
	if step == 0:
		step = 1.0

	for lead_idx, lead_name in enumerate(LEAD_NAMES):
		ax.plot(time, segment[:, lead_idx] + offset, linewidth=0.9)
		ax.text(time[-1] + 5, offset, lead_name, va="center", fontsize=9)
		offset += step

	ax.set_title(title)
	ax.set_xlabel("Campioni")
	ax.set_yticks([])
	ax.spines["top"].set_visible(False)
	ax.spines["right"].set_visible(False)


def pick_first_row(df: pd.DataFrame, label: str) -> pd.Series:
	matches = df[df["label"] == label]
	if matches.empty:
		raise ValueError(f"Nessun segmento trovato con label '{label}'")
	return matches.iloc[0]


def main():
	parser = argparse.ArgumentParser(
		description="Mostra un segmento normale e uno con malposizionamento"
	)
	parser.add_argument(
		"--index_csv",
		default="lead_misplacement/final_dataset_index.csv",
		help="CSV finale prodotto da misplacement_dataset_builder.py",
	)
	parser.add_argument(
		"--segments_dir",
		default="lead_misplacement/segments",
		help="Cartella con i segmenti .npy del dataset finale",
	)
	parser.add_argument(
		"--misplaced_label",
		default="V1_V2",
		help="Etichetta da usare come esempio malposizionato",
	)
	args = parser.parse_args()

	index_df = pd.read_csv(args.index_csv)

	normal_row = pick_first_row(index_df, "normal")
	misplaced_row = pick_first_row(index_df, args.misplaced_label)

	normal_segment = load_segment(args.segments_dir, normal_row["filename"])
	misplaced_segment = load_segment(args.segments_dir, misplaced_row["filename"])

	fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
	plot_segment(
		axes[0],
		normal_segment,
		f"Segmento normale - ecg_id {normal_row['ecg_id']} - label {normal_row['label']}",
	)
	plot_segment(
		axes[1],
		misplaced_segment,
		f"Segmento malposizionato - ecg_id {misplaced_row['ecg_id']} - label {misplaced_row['label']}",
	)

	plt.tight_layout()
	plt.show()


if __name__ == "__main__":
	main()

