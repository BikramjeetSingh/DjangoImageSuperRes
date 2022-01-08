import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WINDOW_SIZE = 8
TILE_SIZE = 400
TILE_OVERLAP = 32
SCALE_FACTOR = 4
# MODEL_FILE = "SwinIR_L_x4_frozen.pth"
MODEL_FILE = "Swin.pth"