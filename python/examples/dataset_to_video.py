"""
Convert a LeRobot dataset's camera images to a video at the real recorded speed.

Usage:
    python dataset_to_video.py --root ./data_bimanual_cam_4
    python dataset_to_video.py --root ./data_bimanual_cam_4 --episode 0 --output video.mp4
"""

import os
import sys
import subprocess
import tempfile

import numpy as np
import pandas as pd
from PIL import Image
import io
import click


def load_episode(root: str, episode: int):
    """Load images and timestamps for a given episode."""
    data_dir = os.path.join(root, "data")
    parquet_files = []
    for dirpath, _, filenames in os.walk(data_dir):
        for fn in sorted(filenames):
            if fn.endswith(".parquet"):
                parquet_files.append(os.path.join(dirpath, fn))

    df = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)
    ep_df = df[df["episode_index"] == episode]
    if len(ep_df) == 0:
        available = sorted(df["episode_index"].unique())
        raise ValueError(f"Episode {episode} not found. Available: {available}")

    if "observation.images.cam" not in ep_df.columns:
        raise ValueError("Dataset has no camera images (observation.images.cam)")

    timestamps = ep_df["timestamp"].values.astype(np.float64)
    images = ep_df["observation.images.cam"].values
    return images, timestamps


@click.command()
@click.option("--root", required=True, help="LeRobot dataset root directory")
@click.option("--episode", default=0, help="Episode index (default: 0)")
@click.option("--output", default=None, help="Output video path (default: <root>/episode_<N>.mp4)")
def main(root: str, episode: int, output: str | None):
    images, timestamps = load_episode(root, episode)
    total = len(images)
    duration = timestamps[-1] - timestamps[0]
    actual_fps = (total - 1) / duration if duration > 0 else 30

    if output is None:
        output = os.path.join(root, f"episode_{episode}.mp4")

    print(f"Episode {episode}: {total} frames, {duration:.1f}s, actual {actual_fps:.1f} fps")

    # Extract frames and build ffmpeg concat list
    with tempfile.TemporaryDirectory() as tmp_dir:
        concat_lines = []
        for i in range(total):
            img = Image.open(io.BytesIO(images[i]["bytes"]))
            frame_path = os.path.join(tmp_dir, f"{i:06d}.png")
            img.save(frame_path)

            if i < total - 1:
                dt = timestamps[i + 1] - timestamps[i]
            else:
                dt = timestamps[i] - timestamps[i - 1]
            concat_lines.append(f"file '{frame_path}'")
            concat_lines.append(f"duration {dt:.6f}")

        concat_path = os.path.join(tmp_dir, "concat.txt")
        with open(concat_path, "w") as f:
            f.write("\n".join(concat_lines) + "\n")

        cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0", "-i", concat_path,
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            output,
        ]
        subprocess.run(cmd, capture_output=True, check=True)

    size_mb = os.path.getsize(output) / 1024 / 1024
    print(f"Saved: {output} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
