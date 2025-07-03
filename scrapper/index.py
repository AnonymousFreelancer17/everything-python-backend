import torch
from deca import DECA
from deca.utils import util

if torch.cuda.is_available():
    print(f"CUDA is available. Using device: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA is not available. Falling back to CPU.")


# Load DECA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
deca = DECA().to(device)

# Load input image
image_path = r'C:/Users/Public/Downloads/ASM/s/rcauntty/aunty.png'
input_image = util.load_image(image_path)

# Generate 3D face
output = deca(input_image)

# Save 3D model
output_path = 'output/head_model.obj'
deca.save_obj(output_path, output)
