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
import mmcv

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)

def generate_connected_blobby_mask(
    shape=(1, 1, 224, 224),
    sigma=10,
    threshold=0.5,
    device='cpu'):    
    h, w = shape
    random_noise = torch.rand((h, w)).numpy()
    smoothed = gaussian_filter(random_noise, sigma=sigma)
    binary_mask = (smoothed > threshold).astype(np.uint8)
    labeled_array, num_features = label(binary_mask)
    if num_features > 0:
        sizes = np.bincount(labeled_array.ravel())
        sizes[0] = 0
        largest_label = sizes.argmax()
        connected_mask = (labeled_array == largest_label).astype(np.float32)
    else:
        connected_mask = np.zeros((h, w), dtype=np.float32)

    mask_tensor = torch.tensor(connected_mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    mask_tensor = mask_tensor.to(device)
    return mask_tensor


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
        self.lower_bound = 9
        self.upper_bound = 12
        
        self.random_augmentor = RandomAugmentations(seed=seed)
        
        self.mean = np.array(img_norm_cfg['mean'], dtype=np.float32)
        self.std = np.array(img_norm_cfg['std'], dtype=np.float32)
        self.to_bgr = img_norm_cfg['to_bgr']
        self.min_speed = 20
        self.max_speed = 40
        
    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def rotate(self, patch, width, height, min_angle=-90, max_angle=90):
        random_rotate = random.uniform(min_angle, max_angle)
        patch = patch.convert("RGBA").rotate(random_rotate, expand=True)
        patch = patch.resize((width, height), resample=Image.BICUBIC)
        mask = patch.split()[-1]
        
        return patch.convert("RGB"), mask

    def intersect_masks(self, mask1, mask2):
        mask1_np = np.array(mask1)
        mask2_np = np.array(mask2)
    
        intersection = np.logical_and(mask1_np, mask2_np).astype(np.uint8) * 255
        intersection_mask = Image.fromarray(intersection)
    
        return intersection_mask
    
    def expand_mask(self, mask, kernel_size=(3, 3)):
        kernel = np.ones(kernel_size, np.uint8)
        expanded_mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=10)
        
        return expanded_mask
    
    def sample_coordinate_shape(self, foreground_mask):
        foreground_mask = self.expand_mask(foreground_mask)
        h, w = foreground_mask.shape
        coords = np.column_stack(np.where(foreground_mask == 1))

        patch_width = random.randint(int(w*0.1), int(w*0.6))
        patch_height =  random.randint(int(h*0.1), int(h*0.6))

        y1, x1 = coords[random.randint(0, len(coords) - 1)]
        y2 = random.randint(0, h - patch_height - 2)
        x2 = random.randint(0, w - patch_width - 2)

        return x1, y1, x2, y2, patch_width, patch_height
    
    def __call__(self, imgs):
        t, c, h, w = imgs.shape

        angle = np.random.rand() * 2 * np.pi
        speed = np.random.uniform(self.min_speed, self.max_speed)
        dx, dy = speed * np.cos(angle), speed * np.sin(angle)

        fg_mask = generate_connected_blobby_mask((h, w), sigma=14, threshold=0.5)
        x1_0, y1_0, _, _, patch_w, patch_h = self.sample_coordinate_shape(fg_mask.cpu().squeeze().numpy())

        transformed = []
        for i in range(t):
            x1 = int(x1_0 + dx * i)
            y1 = int(y1_0 + dy * i)
            x2 = x1 + int(patch_w)
            y2 = y1 + int(patch_h)

            x1 = np.clip(x1, 0, w - int(patch_w))
            y1 = np.clip(y1, 0, h - int(patch_h))
            x2, y2 = x1 + int(patch_w), y1 + int(patch_h)

            frame = imgs[i].cpu().float().numpy()
            frame = mmcv.imdenormalize(frame.transpose(1,2,0), self.mean, self.std, self.to_bgr)
            pil = transforms.ToPILImage()(frame)

            # crop the moving patch out of the SAME static location on the FIRST frame—
            # or you could crop each frame, but usually you want a single patch image.
            if i == 0:
                patch = pil.crop((x1, y1, x2, y2))
                patch = self.random_augmentor.apply(patch,
                            np.random.choice(['light','medium','heavy'], p=[0.3,0.4,0.3]))
                patch, rotation_mask = self.rotate(patch, patch_w, patch_h)
                mask = np.ones((patch_h, patch_w), dtype=np.uint8)
                mask = cv2.resize(mask, (patch_w, patch_h), interpolation=cv2.INTER_CUBIC)
                mask = self.intersect_masks(mask, rotation_mask)

            augmented = pil.copy()
            augmented.paste(patch, (x1, y1), mask=mask)

            out = np.array(augmented) * 255
            out = mmcv.imnormalize(out, self.mean, self.std, self.to_bgr)
            transformed.append(transforms.ToTensor()(out))

        return torch.stack(transformed)