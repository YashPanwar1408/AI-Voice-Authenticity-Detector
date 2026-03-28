"""
generate_spectrograms.py
------------------------
Converts .wav audio files into mel spectrogram images (.png).

Folder structure:
    Input  → data/audio/REAL  and  data/audio/FAKE
    Output → data/spectrograms/REAL  and  data/spectrograms/FAKE

Run:
    python -m src.generate_spectrograms
"""

import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")   # headless backend — no display needed


# ── Config ─────────────────────────────────────────────────────────────────────
AUDIO_ROOT   = "data/audio"
OUTPUT_ROOT  = "data/spectrograms"
CLASSES      = ["REAL", "FAKE"]
SAMPLE_RATE  = 16000
IMG_SIZE     = 128          # pixels (width == height)
N_MELS       = 128          # mel filter banks
# ───────────────────────────────────────────────────────────────────────────────


def save_spectrogram(audio_path: str, output_path: str) -> bool:
    """
    Load one .wav file, compute a mel spectrogram in dB, and save as PNG.

    Args:
        audio_path:  Full path to the source .wav file.
        output_path: Full path for the output .png image.

    Returns:
        True on success, False on error.
    """
    try:
        # Load audio
        audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE)

        # Mel spectrogram → dB scale
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=N_MELS)
        mel_db   = librosa.power_to_db(mel_spec, ref=np.max)

        # Render clean image (no axes, no border, exact pixel size)
        dpi = 100
        fig_size = IMG_SIZE / dpi   # inches

        fig, ax = plt.subplots(1, 1, figsize=(fig_size, fig_size), dpi=dpi)
        ax.imshow(mel_db, aspect="auto", origin="lower", cmap="magma")
        ax.axis("off")
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

        fig.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0)
        plt.close(fig)

        return True

    except Exception as e:
        print(f"  [ERROR] {audio_path}: {e}")
        return False


def generate_spectrograms(
    audio_root:  str = AUDIO_ROOT,
    output_root: str = OUTPUT_ROOT,
) -> None:
    """
    Iterate through REAL/ and FAKE/ sub-folders, convert every .wav to a
    mel spectrogram PNG, and mirror the class structure in the output folder.

    Args:
        audio_root:  Root folder containing REAL/ and FAKE/ sub-folders.
        output_root: Root folder where spectrogram images are saved.
    """
    total_ok    = 0
    total_fail  = 0

    for class_name in CLASSES:
        input_dir  = os.path.join(audio_root,  class_name)
        output_dir = os.path.join(output_root, class_name)

        if not os.path.isdir(input_dir):
            print(f"[WARNING] Input folder not found, skipping: {input_dir}")
            continue

        os.makedirs(output_dir, exist_ok=True)

        wav_files = sorted(
            f for f in os.listdir(input_dir) if f.lower().endswith(".wav")
        )
        total = len(wav_files)
        print(f"\n[{class_name}]  {total} files  →  {output_dir}")

        ok = fail = 0

        for i, filename in enumerate(wav_files, start=1):
            audio_path  = os.path.join(input_dir, filename)
            output_name = os.path.splitext(filename)[0] + ".png"
            output_path = os.path.join(output_dir, output_name)

            if save_spectrogram(audio_path, output_path):
                ok += 1
            else:
                fail += 1

            if i % 100 == 0 or i == total:
                print(f"  {i}/{total}  (ok={ok}, fail={fail})")

        total_ok   += ok
        total_fail += fail

    print(f"\n{'─' * 42}")
    print(f"  Done.  Saved: {total_ok}  |  Failed: {total_fail}")
    print(f"  Output root → {os.path.abspath(output_root)}")
    print(f"{'─' * 42}")


if __name__ == "__main__":
    generate_spectrograms()
