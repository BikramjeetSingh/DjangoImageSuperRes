import torch
import numpy as np
import os
import boto3
from PIL import Image

from upscaler import config


def _get_model():
    if not os.path.exists(os.path.join("upscaler", "models", config.MODEL_FILE)):
        s3 = boto3.client('s3')
        s3.download_file('imagesrbucket', 'SwinIR_L_x4_frozen.pth', os.path.join("upscaler", "models", config.MODEL_FILE))



def _add_padding(_img, _window_size):
    _, height, _ = _img.shape
    height_padding = (height // _window_size + 1) * _window_size - height
    height_pad_pixels = _img[:, :height_padding, :]
    height_pad_pixels = np.flip(height_pad_pixels, axis=1)
    image_padded = np.concatenate([height_pad_pixels, _img], axis=1)

    _, _, width = image_padded.shape
    width_padding = (width // _window_size + 1) * _window_size - width
    width_pad_pixels = image_padded[:, :, :width_padding]
    width_pad_pixels = np.flip(width_pad_pixels, axis=2)
    image_padded = np.concatenate([width_pad_pixels, image_padded], axis=2)

    return image_padded


def _perform_upscaling(_img, _model, _tile_size, _tile_overlap, _scale_factor):
    stride = _tile_size - _tile_overlap

    b, c, h, w = _img.size()

    h_idx_list = list(range(0, h - _tile_size, stride)) + [h - _tile_size]
    w_idx_list = list(range(0, w - _tile_size, stride)) + [w - _tile_size]
    scaled = torch.zeros(b, c, h * _scale_factor, w * _scale_factor).float().to(config.DEVICE)
    overlap = torch.zeros(b, c, h * _scale_factor, w * _scale_factor).float().to(config.DEVICE)

    for h_idx in h_idx_list:
        for w_idx in w_idx_list:
            in_patch = _img[:, :, h_idx:h_idx + _tile_size, w_idx:w_idx + _tile_size]
            out_patch = _model(in_patch)
            out_patch_mask = torch.ones_like(out_patch)

            scaled[
                :, :,
                h_idx * _scale_factor:(h_idx + _tile_size) * _scale_factor,
                w_idx * _scale_factor:(w_idx + _tile_size) * _scale_factor,
            ].add_(out_patch)

            overlap[
                :, :,
                h_idx * _scale_factor:(h_idx + _tile_size) * _scale_factor,
                w_idx * _scale_factor:(w_idx + _tile_size) * _scale_factor,
            ].add_(out_patch_mask)

    output = scaled.div_(overlap)
    return output


def predict(img_path: str, output_path: str):
    _get_model()
    model = torch.load(os.path.join("upscaler", "models", config.MODEL_FILE))
    model = model.to(config.DEVICE)

    img = Image.open(img_path)
    img = np.array(img) / 255.0
    img = np.moveaxis(img, -1, 0)

    _, height, width = img.shape

    img = _add_padding(img, config.WINDOW_SIZE)

    img = torch.from_numpy(img).float().unsqueeze(0).to(config.DEVICE)

    with torch.no_grad():
        output = _perform_upscaling(img, model, config.TILE_SIZE, config.TILE_OVERLAP, config.SCALE_FACTOR)
        output = output[:, :, :height * config.SCALE_FACTOR, :width * config.SCALE_FACTOR]

    output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
    output = (output * 255.0).round().astype(np.uint8)
    output = np.moveaxis(output, 0, -1)
    Image.fromarray(output).save(output_path)
    return output_path
