#%%
import os
import imageio
import glob
import cv2
import numpy as np
from PIL import Image

def video_to_images(video_path):
    """Parses a video into individual image frames."""

    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    count = 0

    images = [image]

    while success:
        success, image = vidcap.read()
        count += 1
        if image is not None:
            images.append(image)

    vidcap.release()

    return images

def rgba_to_rgb(rgba, background=(1., 1., 1.)):
    """Converts RGBA to RGB with alpha blending.

    Args:
        rgba (tuple): RGBA color tuple (r, g, b, a).
        background (tuple, optional): Background color tuple (r, g, b). Defaults to white (255, 255, 255).

    Returns:
        tuple: RGB color tuple (r, g, b).
    """

    rgb = rgba[..., :3]
    alpha = rgba[..., 3]
    bg_r, bg_g, bg_b = background

    return rgb * alpha[..., None] + np.array(background)[None, None] * (1. - alpha[..., None])

file = 'videos/closed-loop/IMG_2624_0'
images = video_to_images(file + '.MOV')
images_mask = video_to_images(file + '_masked.mp4')

os.makedirs(file, exist_ok=True)

background = cv2.resize(images[0], (1280, 720))
# Creating kernel 
kernel = np.ones((5, 5), np.uint8) 

drone_masks = []
for i, (img) in enumerate(images_mask[::3]):
    img_black_mask = np.linalg.norm(img, axis=-1) < 50
    img_black_mask = cv2.dilate(img_black_mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    # Apply Gaussian blur
    smoothed_mask = cv2.GaussianBlur( (~img_black_mask).astype(np.uint8), (7, 7), 0)

    img_black_mask = smoothed_mask < 0.6

    alpha = (255*np.ones(img.shape[:2])).astype(np.uint8)
    alpha[img_black_mask] = 0
    img[img_black_mask] = 255
    full_image = np.concatenate([img, alpha[:, :, np.newaxis]], axis=-1)
    bgr_image = cv2.cvtColor(full_image, cv2.COLOR_RGBA2BGR)

    background[~img_black_mask] = img[~img_black_mask]
    drone_masks.append(~img_black_mask)

drone_masks_img = np.stack(drone_masks, axis=0)
drone_masks_img = np.any(drone_masks_img, axis=0)

alpha = np.zeros(background.shape[:2], dtype=np.uint8)
for mask_ in drone_masks:
    alpha = alpha + (255*mask_).astype(np.uint8)
alpha = alpha / len(drone_masks)
alpha = alpha / alpha.max()
alpha[~drone_masks_img] = 1.

alpha = (255*alpha).astype(np.uint8)

background = cv2.cvtColor(background, cv2.COLOR_RGB2BGR)
background = np.concatenate([background, alpha[:, :, np.newaxis]], axis=-1)

background = background / 255.
background = rgba_to_rgb(background)
background = (255*background).astype(np.uint8)

background_img = Image.fromarray(background, 'RGB')
background_img.save(f'{file.split(".")[0]}/blended.png')

#%%