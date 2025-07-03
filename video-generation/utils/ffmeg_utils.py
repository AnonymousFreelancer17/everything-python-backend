import os
import subprocess
from typing import List
from PIL import Image

def save_frames_as_video(frames: List[Image.Image], output_path: str, fps=8):
    tmp_dir = "./temp_frames"
    os.makedirs(tmp_dir, exist_ok=True)

    for i, frame in enumerate(frames):
        frame.save(f"{tmp_dir}/frame_{i:03d}.png")

    subprocess.call([
        "ffmpeg", "-y", "-framerate", str(fps), "-i", f"{tmp_dir}/frame_%03d.png",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", output_path
    ])

    for f in os.listdir(tmp_dir):
        os.remove(os.path.join(tmp_dir, f))
    os.rmdir(tmp_dir)
