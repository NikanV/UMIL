import cv2
import torch
import noise
import random
import numpy as np
from PIL import Image
import albumentations as A
from torchvision import transforms
from scipy.ndimage import gaussian_filter
from scipy.ndimage import gaussian_filter, map_coordinates
import torch
import numpy as np
from scipy.ndimage import gaussian_filter, label
import matplotlib.pyplot as plt

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)

class RandomAugmentations:
    def __init__(self, seed=None):
        self.seed = seed
        # self.set_seed(seed)

        self.color_transform_light = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
        self.color_transform_medium = transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2)
        self.color_transform_heavy = transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.3)

        self.albumentations_color = A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.5, contrast_limit=0.5, p=0.8),
            A.HueSaturationValue(hue_shift_limit=50, sat_shift_limit=50, val_shift_limit=50, p=0.8),
            A.RGBShift(r_shift_limit=30, g_shift_limit=30, b_shift_limit=30, p=0.8),
        ])

        self.augmentations = [
            self.elastic_transform, self.salt_and_pepper_noise, self.torn_paper_effect,
            self.color_transformation, 
            self.swirl_distortion, 
            self.gaussian_blur
        ]

    def set_seed(self, seed):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    def apply(self, image, level='medium'):
        image_np = np.array(image)
        
        if level == 'light':
            n_augmentations = random.randint(2, 3)
        elif level == 'medium':
            n_augmentations = random.randint(3, 6)
        else:  # heavy
            n_augmentations = random.randint(6, len(self.augmentations))

        selected_augmentations = random.sample(self.augmentations, n_augmentations)

        for augmentation in selected_augmentations:
            image_np = augmentation(image_np, level)

        return Image.fromarray(image_np)

    def elastic_transform(self, image, level, alpha=None, sigma=None):
        alpha = alpha or {'light': 20, 'medium': 40, 'heavy': 60}[level]
        sigma = sigma or {'light': 2, 'medium': 4, 'heavy': 6}[level]
        
        random_state = np.random.RandomState(self.seed)
        shape = image.shape

        dx = gaussian_filter((random_state.rand(*shape[:2]) * 2 - 1), sigma, mode="reflect") * alpha
        dy = gaussian_filter((random_state.rand(*shape[:2]) * 2 - 1), sigma, mode="reflect") * alpha

        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        indices = (y + dy).flatten(), (x + dx).flatten()

        distorted_image = np.zeros_like(image)

        for i in range(shape[2]):
            distorted_image[..., i] = map_coordinates(image[..., i], indices, order=1, mode='reflect').reshape(shape[:2])

        return distorted_image

    def salt_and_pepper_noise(self, image, level, salt_prob=None, pepper_prob=None):
        salt_prob = salt_prob or {'light': 0.01, 'medium': 0.05, 'heavy': 0.1}[level]
        pepper_prob = pepper_prob or {'light': 0.01, 'medium': 0.05, 'heavy': 0.1}[level]

        image_np = image.copy()
        total_pixels = image_np.size
        num_salt = np.ceil(salt_prob * total_pixels)
        num_pepper = np.ceil(pepper_prob * total_pixels)

        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image_np.shape]
        image_np[coords[0], coords[1]] = 255

        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image_np.shape]
        image_np[coords[0], coords[1]] = 0

        return image_np

    def torn_paper_effect(self, image, level):
        image_np = image.copy()
        height, width = image_np.shape[:2]

        num_lines = {'light': 5, 'medium': 10, 'heavy': 20}[level]
        for _ in range(num_lines):
            start_x = np.random.randint(0, width)
            start_y = np.random.randint(0, height)
            end_x = np.random.randint(0, width)
            end_y = np.random.randint(0, height)
            cv2.line(image_np, (start_x, start_y), (end_x, end_y), (255, 255, 255), thickness=1)

        return image_np

    def perlin_noise_mask(self, image, level, scale=None):
        scale = scale or {'light': 20, 'medium': 10, 'heavy': 5}[level]

        height, width = image.shape[:2]
        mask = np.zeros((height, width), dtype=np.float32)

        for i in range(height):
            for j in range(width):
                mask[i, j] = noise.pnoise2(i / scale, j / scale, octaves=6)

        mask = (mask - mask.min()) / (mask.max() - mask.min()) * 255
        image[mask > 128] = np.random.randint(0, 255, 3)

        return image

    def color_transformation(self, image, level):
        transform = {'light': self.color_transform_light, 'medium': self.color_transform_medium, 'heavy': self.color_transform_heavy}[level]
        return np.array(transform(Image.fromarray(image)))

    def swirl_distortion(self, image, level, strength=None):
        strength = strength or {'light': 1, 'medium': 3, 'heavy': 5}[level]
        patch_np = np.array(image)

        height, width = patch_np.shape[:2]
        center_x, center_y = width // 2, height // 2

        y, x = np.indices((height, width))
        x = x - center_x
        y = y - center_y
        distance = np.sqrt(x**2 + y**2)

        angle = strength * np.exp(-distance**2 / (2 * (min(height, width) // 3)**2))

        new_x = center_x + x * np.cos(angle) - y * np.sin(angle)
        new_y = center_y + x * np.sin(angle) + y * np.cos(angle)

        map_x = np.clip(new_x, 0, width - 1).astype(np.float32)
        map_y = np.clip(new_y, 0, height - 1).astype(np.float32)

        return cv2.remap(patch_np, map_x, map_y, interpolation=cv2.INTER_LINEAR)

    def gaussian_blur(self, image, level):
        kernel_size = {'light': 3, 'medium': 5, 'heavy': 7}[level]
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

class AnomalyGenerator(object):
    def __init__(self, seed=None):
        self.mean = np.array(img_norm_cfg['mean'], dtype=np.float32)
        self.std = np.array(img_norm_cfg['std'], dtype=np.float32)
        self.to_bgr = img_norm_cfg['to_bgr']
        
        self.random = random.Random(seed)
        self.min_speed = 20
        self.max_speed = 35

        self.augmentor = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
    
    def generate_blob_mask(self, height, width, sigma=14, threshold=0.5, device='cpu'):
        # create smooth random noise and threshold to single largest connected region
        rnd = torch.rand((height, width)).numpy()
        sm = cv2.GaussianBlur(rnd, (0, 0), sigmaX=sigma, sigmaY=sigma)
        binary = (sm > threshold).astype(np.uint8)
        num_labels, labels = cv2.connectedComponents(binary)
        if num_labels > 1:
            # pick largest blob (skip background label=0)
            sizes = [(labels == i).sum() for i in range(1, num_labels)]
            largest = np.argmax(sizes) + 1
            mask = (labels == largest).astype(np.uint8)
        else:
            mask = np.zeros((height, width), dtype=np.uint8)
        return mask
    
    def __call__(self, imgs):
        """
        imgs: tensor [T, C, H, W], float in [0,1]
        returns: tensor [T, C, H, W], float in [0,1]
        """
        T, C, H, W = imgs.shape
        # random movement vector
        angle = self.random.uniform(0, 2 * np.pi)
        speed = self.random.uniform(self.min_speed, self.max_speed)
        dx, dy = speed * np.cos(angle), speed * np.sin(angle)

        # generate mask and choose patch size
        mask = self.generate_blob_mask(H, W)
        pw = self.random.randint(int(W * 0.1), int(W * 0.5))
        ph = self.random.randint(int(H * 0.1), int(H * 0.5))

        # pick a patch center near image center with small jitter
        cx, cy = W // 2, H // 2
        max_jitter_x = int(W * 0.1)
        max_jitter_y = int(H * 0.1)
        jitter_x = self.random.randint(-max_jitter_x, max_jitter_x)
        jitter_y = self.random.randint(-max_jitter_y, max_jitter_y)
        x0 = np.clip(cx - pw // 2 + jitter_x, 0, W - pw)
        y0 = np.clip(cy - ph // 2 + jitter_y, 0, H - ph)

        # extract patch from first frame
        frame0 = imgs[0].cpu().numpy().transpose(1, 2, 0)
        pil0 = transforms.ToPILImage()(frame0)
        patch = pil0.crop((x0, y0, x0 + pw, y0 + ph))
        patch = self.augmentor(patch)
        patch = patch.convert('RGBA')
        alpha = np.array(patch.split()[-1])
        patch_rgb = np.array(patch.convert('RGB'))

        outputs = []
        for t in range(T):
            xi = int(np.clip(x0 + dx * t, 0, W - pw))
            yi = int(np.clip(y0 + dy * t, 0, H - ph))

            frame = imgs[t].cpu().numpy().transpose(1, 2, 0)
            pil_frame = transforms.ToPILImage()(frame).convert('RGBA')
            pil_frame.paste(Image.fromarray(patch_rgb), (xi, yi), mask=Image.fromarray(alpha))

            out_np = np.array(pil_frame.convert('RGB'))
            out_tensor = transforms.ToTensor()(out_np)
            outputs.append(out_tensor)

        return torch.stack(outputs)