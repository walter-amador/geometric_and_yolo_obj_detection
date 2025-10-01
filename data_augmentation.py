import cv2
import numpy as np
import random
import os


class ImageTransformer:
    def __init__(self, image_path):
        self.image = cv2.imread(image_path)
        self.image = self.resize_image()

    def resize_image(self, width=640, height=480):
        h, w = self.image.shape[:2]
        scale = min(width / w, height / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(self.image, (new_w, new_h))
        # Pad to target size
        pad_w = (width - new_w) // 2
        pad_h = (height - new_h) // 2
        padded = cv2.copyMakeBorder(
            resized,
            pad_h,
            height - new_h - pad_h,
            pad_w,
            width - new_w - pad_w,
            cv2.BORDER_CONSTANT,
            value=[0, 0, 0],
        )
        return padded


class GeometricTransformer:

    def __init__(self, image):
        self.image = image

    def apply_transformations(
        self,
        rotation_angles=[30, -30],
        shear_factors=[0.2, -0.2],
        translations=[(-10, 0), (10, 0), (0, -10), (0, 10)],
        scale_factors=[0.8, 1.2],
    ):
        """Apply a series of transformations to the image using parameter lists for each type."""
        transformed_images = []

        # Apply rotation for each angle in the list
        for angle in rotation_angles:
            transformed_images.append(self._rotate(angle))

        # Apply shearing for each shear factor in the list
        for shear in shear_factors:
            transformed_images.append(self._shear(shear))

        # Apply translation for each (tx, ty) tuple in the list
        for tx, ty in translations:
            transformed_images.append(self._translate(tx, ty, percent=True))

        # Apply scaling for each scale factor in the list
        for scale in scale_factors:
            transformed_images.append(self._scale(scale))

        return transformed_images

    def _rotate(self, angle=30):
        """Rotate the image by the specified angle."""
        (h, w) = self.image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(self.image, M, (w, h))
        return rotated

    def _shear(self, shear_factor=0.1):
        """Shear the image by the specified shear factor."""
        (h, w) = self.image.shape[:2]
        M = np.array([[1, shear_factor, 0], [0, 1, 0]], dtype=np.float32)
        sheared = cv2.warpAffine(self.image, M, (w, h))
        return sheared

    def _translate(self, tx=20, ty=20, percent=False):
        """Translate the image by the specified x and y offsets.
        If percent=True, tx and ty are interpreted as percentages of width and height.
        """
        (h, w) = self.image.shape[:2]
        if percent:
            tx = int(w * tx / 100.0)
            ty = int(h * ty / 100.0)
        M = np.array([[1, 0, tx], [0, 1, ty]], dtype=np.float32)
        translated = cv2.warpAffine(self.image, M, (w, h))
        return translated

    def _scale(self, scale_factor=1.2):
        """Scale the image by the specified scale factor, then crop or pad to 640x480."""
        (h, w) = self.image.shape[:2]
        new_w, new_h = int(w * scale_factor), int(h * scale_factor)
        scaled = cv2.resize(self.image, (new_w, new_h))

        target_w, target_h = 640, 480

        if new_w > target_w or new_h > target_h:
            # Crop the center
            start_x = (new_w - target_w) // 2
            start_y = (new_h - target_h) // 2
            cropped = scaled[start_y : start_y + target_h, start_x : start_x + target_w]
            return cropped
        else:
            # Pad the image to the target size
            pad_x = (target_w - new_w) // 2
            pad_y = (target_h - new_h) // 2
            padded = cv2.copyMakeBorder(
                scaled,
                pad_y,
                target_h - new_h - pad_y,
                pad_x,
                target_w - new_w - pad_x,
                cv2.BORDER_CONSTANT,
                value=[0, 0, 0],
            )
            return padded


class PhotometricTransformer:
    def __init__(self, image):
        self.image = image

    def random_transform(self):
        img = self.image.copy()
        transforms = [
            self._change_brightness,
            self._jitter,
            self._hsv_shift,
            self._to_grayscale,
            self._add_noise,
        ]
        # Randomly select 1-3 transformations to apply
        num_transforms = random.randint(1, 3)
        selected = random.sample(transforms, num_transforms)
        for t in selected:
            img = t(img)
        return img

    def _change_brightness(self, img, factor_range=(0.6, 1.4)):
        factor = random.uniform(*factor_range)
        img = cv2.convertScaleAbs(img, alpha=factor, beta=0)
        return img

    def _jitter(self, img, jitter_range=20):
        # Add random value to each channel
        jitter = np.random.randint(
            -jitter_range, jitter_range + 1, img.shape, dtype=np.int16
        )
        img = img.astype(np.int16) + jitter
        img = np.clip(img, 0, 255).astype(np.uint8)
        return img

    def _hsv_shift(self, img, h_shift=20, s_shift=30, v_shift=30):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)
        h = (hsv[..., 0] + random.randint(-h_shift, h_shift)) % 180
        s = np.clip(hsv[..., 1] + random.randint(-s_shift, s_shift), 0, 255)
        v = np.clip(hsv[..., 2] + random.randint(-v_shift, v_shift), 0, 255)
        hsv_shifted = np.stack([h, s, v], axis=-1).astype(np.uint8)
        img = cv2.cvtColor(hsv_shifted, cv2.COLOR_HSV2BGR)
        return img

    def _to_grayscale(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    def _add_noise(self, img, mean=0, std=15):
        noise = np.random.normal(mean, std, img.shape).astype(np.int16)
        img = img.astype(np.int16) + noise
        img = np.clip(img, 0, 255).astype(np.uint8)
        return img


raw_data_dir = "raw_data"
resized_dir = "resized"
os.makedirs(resized_dir, exist_ok=True)

for filename in os.listdir(raw_data_dir):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        image_path = os.path.join(raw_data_dir, filename)
        image_transformer = ImageTransformer(image_path)
        image = image_transformer.image
        save_path = os.path.join(resized_dir, filename)
        cv2.imwrite(save_path, image)

# geo_transformer = GeometricTransformer(image)
# transformed_images = geo_transformer.apply_transformations()

# for idx, img in enumerate(transformed_images):
#     photo_transformer = PhotometricTransformer(img)
#     transformed_images[idx] = photo_transformer.random_transform()

# # save transformed images to augmented_data directory
# for idx, img in enumerate(transformed_images):
#     cv2.imwrite(f"augmented_data/transformed_{idx}.jpeg", img)
