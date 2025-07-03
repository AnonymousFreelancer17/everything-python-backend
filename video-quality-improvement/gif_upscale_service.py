from fastapi import FastAPI, UploadFile, File
import os
import imageio
from PIL import Image
import subprocess
from pathlib import Path

app = FastAPI()
UPLOAD_DIR = "uploaded_gifs"
UPSCALED_DIR = "upscaled_gifs"
MODEL_NAME = "RealESRGAN_x4plus"  # x4 upscaling (ideal for 2K)

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(UPSCALED_DIR, exist_ok=True)

@app.post("/upscale-gif")
async def upscale_gif(file: UploadFile = File(...), resolution: str = "2k"):
    input_path = f"{UPLOAD_DIR}/{file.filename}"
    with open(input_path, "wb") as f:
        f.write(await file.read())

    # Step 1: Extract frames
    frames = imageio.mimread(input_path)
    temp_frame_dir = Path("frames")
    temp_frame_dir.mkdir(exist_ok=True)
    for i, frame in enumerate(frames):
        Image.fromarray(frame).save(temp_frame_dir / f"frame_{i:03d}.png")

    # Step 2: Upscale using Real-ESRGAN CLI
    output_frame_dir = Path("frames_upscaled")
    output_frame_dir.mkdir(exist_ok=True)

    subprocess.run([
        "python", "inference_realesrgan.py",
        "-n", MODEL_NAME,
        "-i", str(temp_frame_dir),
        "-o", str(output_frame_dir),
        "--face_enhance"
    ])

    # Step 3: Rebuild GIF
    upscaled_frames = sorted(output_frame_dir.glob("*.png"))
    upscaled_images = [imageio.imread(str(img)) for img in upscaled_frames]
    output_gif_path = f"{UPSCALED_DIR}/upscaled_{file.filename}"

    imageio.mimsave(output_gif_path, upscaled_images, duration=0.04)

    return {"message": "GIF upscaled successfully", "output_path": output_gif_path}
