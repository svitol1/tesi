"""Plot a single hardcoded ECG segment from georgia_dataset/segments."""

from pathlib import Path

import numpy as np
import ecg_plot


def main():
    segment_path = Path("georgia_dataset") / "segments" / "seg000092.npy"
    segment = np.load(segment_path).astype(np.float32)

    # ecg_plot.plot expects an array shaped like (n_leads, n_samples)
    ecg_plot.plot(segment, sample_rate=500, title="Georgia ECG segment")
    ecg_plot.show()


if __name__ == "__main__":
    main()