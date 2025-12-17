"""
Image Input Handler Component
Handles loading and preprocessing of images for sketch-to-flowchart conversion
Uses Gemini Vision API as primary processing method
"""

import logging
import asyncio
import numpy as np
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
from PIL import Image, ImageEnhance, ImageOps
import io

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logging.warning("opencv-python not available. Advanced preprocessing features disabled.")


class ImageInputHandler:
    """Handle image input and preprocessing for sketch-to-flowchart conversion"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Preprocessing parameters
        self.target_size = tuple(config.get('image_target_size', [1024, 1024]))
        self.enable_rotation_correction = config.get('enable_rotation_correction', True)
        self.enable_contrast_enhancement = config.get('enable_contrast_enhancement', True)
        self.enable_noise_reduction = config.get('enable_noise_reduction', False)
        self.preprocessing_level = config.get('preprocessing_level', 'auto')

        self.logger.info("ImageInputHandler initialized")
        self.logger.info(f"  Target size: {self.target_size}")
        self.logger.info(f"  Rotation correction: {self.enable_rotation_correction}")
        self.logger.info(f"  Contrast enhancement: {self.enable_contrast_enhancement}")
        self.logger.info(f"  OpenCV available: {CV2_AVAILABLE}")

    async def load_and_preprocess(self, input_path: str) -> Image.Image:
        """
        Load image and apply preprocessing pipeline

        Args:
            input_path: Path to image file (PNG, JPG, JPEG, PDF)

        Returns:
            Preprocessed PIL Image ready for Gemini Vision API
        """
        try:
            # Load image
            image = await asyncio.to_thread(self._load_image, input_path)

            if image is None:
                raise ValueError(f"Failed to load image from {input_path}")

            self.logger.info(f"Loaded image: {image.size}, mode: {image.mode}")

            # Apply preprocessing based on configuration
            if self.preprocessing_level == 'none':
                # No preprocessing, just resize
                image = self._resize_for_vision_api(image)
                return image

            # Auto or manual preprocessing
            if self.enable_rotation_correction and CV2_AVAILABLE:
                image = await asyncio.to_thread(self._correct_rotation, image)

            if self.enable_contrast_enhancement:
                image = await asyncio.to_thread(self._enhance_contrast, image)

            if self.enable_noise_reduction and CV2_AVAILABLE:
                image = await asyncio.to_thread(self._reduce_noise, image)

            # Always resize to optimal size for API
            image = self._resize_for_vision_api(image)

            self.logger.info(f"Preprocessing complete: {image.size}")
            return image

        except Exception as e:
            self.logger.error(f"Image preprocessing failed: {e}")
            raise

    def _load_image(self, path: str) -> Optional[Image.Image]:
        """
        Load image from file path

        Supports: PNG, JPG, JPEG, PDF (first page)
        """
        try:
            path_obj = Path(path)

            if not path_obj.exists():
                self.logger.error(f"Image file not found: {path}")
                return None

            # Handle PDF files (extract first page)
            if path_obj.suffix.lower() == '.pdf':
                return self._load_pdf_page(path)

            # Load regular image files
            image = Image.open(path)

            # Convert to RGB if needed
            if image.mode not in ('RGB', 'L'):
                image = image.convert('RGB')

            return image

        except Exception as e:
            self.logger.error(f"Failed to load image: {e}")
            return None

    def _load_pdf_page(self, pdf_path: str) -> Optional[Image.Image]:
        """
        Load first page of PDF as image

        Requires: pdf2image library
        """
        try:
            from pdf2image import convert_from_path

            images = convert_from_path(pdf_path, first_page=1, last_page=1)
            if images:
                return images[0]
            else:
                self.logger.error("PDF has no pages")
                return None

        except ImportError:
            self.logger.warning("pdf2image not available. Install with: pip install pdf2image")
            return None
        except Exception as e:
            self.logger.error(f"Failed to load PDF: {e}")
            return None

    def _correct_rotation(self, image: Image.Image) -> Image.Image:
        """
        Detect and correct image rotation

        Uses OpenCV to detect text orientation and rotate accordingly
        """
        if not CV2_AVAILABLE:
            self.logger.debug("OpenCV not available, skipping rotation correction")
            return image

        try:
            # Convert PIL to OpenCV format
            cv_image = np.array(image)
            if len(cv_image.shape) == 2:  # Grayscale
                gray = cv_image
            else:  # RGB
                gray = cv2.cvtColor(cv_image, cv2.COLOR_RGB2GRAY)

            # Detect rotation angle using text orientation
            angle = self._detect_rotation_angle(gray)

            if abs(angle) > 0.5:  # Only rotate if angle is significant
                self.logger.info(f"Detected rotation: {angle:.1f} degrees")

                # Rotate image
                (h, w) = cv_image.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated = cv2.warpAffine(cv_image, M, (w, h),
                                        flags=cv2.INTER_CUBIC,
                                        borderMode=cv2.BORDER_REPLICATE)

                # Convert back to PIL
                return Image.fromarray(rotated)
            else:
                self.logger.debug("No significant rotation detected")
                return image

        except Exception as e:
            self.logger.warning(f"Rotation correction failed: {e}")
            return image

    def _detect_rotation_angle(self, gray: np.ndarray) -> float:
        """
        Detect rotation angle using Hough Line Transform

        Returns angle in degrees
        """
        try:
            # Edge detection
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)

            # Detect lines
            lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)

            if lines is None:
                return 0.0

            # Calculate average angle
            angles = []
            for line in lines[:50]:  # Use first 50 lines
                rho, theta = line[0]
                angle = theta * 180 / np.pi - 90
                # Only consider near-horizontal and near-vertical lines
                if abs(angle) < 45 or abs(angle) > 135:
                    angles.append(angle)

            if not angles:
                return 0.0

            # Return median angle
            return float(np.median(angles))

        except Exception as e:
            self.logger.warning(f"Angle detection failed: {e}")
            return 0.0

    def _enhance_contrast(self, image: Image.Image) -> Image.Image:
        """
        Enhance image contrast for better recognition

        Uses CLAHE if OpenCV available, otherwise simple enhancement
        """
        try:
            if CV2_AVAILABLE:
                # Use CLAHE (Contrast Limited Adaptive Histogram Equalization)
                return self._apply_clahe(image)
            else:
                # Simple contrast enhancement using PIL
                enhancer = ImageEnhance.Contrast(image)
                return enhancer.enhance(1.5)  # 50% more contrast

        except Exception as e:
            self.logger.warning(f"Contrast enhancement failed: {e}")
            return image

    def _apply_clahe(self, image: Image.Image) -> Image.Image:
        """
        Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        """
        try:
            # Convert to OpenCV format
            cv_image = np.array(image)

            # Convert to LAB color space
            if len(cv_image.shape) == 3:
                lab = cv2.cvtColor(cv_image, cv2.COLOR_RGB2LAB)
                l, a, b = cv2.split(lab)
            else:
                l = cv_image
                a, b = None, None

            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)

            # Merge channels and convert back
            if a is not None:
                limg = cv2.merge((cl, a, b))
                final = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
            else:
                final = cl

            return Image.fromarray(final)

        except Exception as e:
            self.logger.warning(f"CLAHE failed: {e}")
            return image

    def _reduce_noise(self, image: Image.Image) -> Image.Image:
        """
        Reduce image noise using OpenCV

        Uses Non-local Means Denoising
        """
        if not CV2_AVAILABLE:
            return image

        try:
            # Convert to OpenCV format
            cv_image = np.array(image)

            # Apply denoising
            if len(cv_image.shape) == 3:
                denoised = cv2.fastNlMeansDenoisingColored(cv_image, None, 10, 10, 7, 21)
            else:
                denoised = cv2.fastNlMeansDenoising(cv_image, None, 10, 7, 21)

            return Image.fromarray(denoised)

        except Exception as e:
            self.logger.warning(f"Noise reduction failed: {e}")
            return image

    def _resize_for_vision_api(self, image: Image.Image) -> Image.Image:
        """
        Resize image to optimal size for Gemini Vision API

        Maintains aspect ratio, max dimension 1024 pixels
        """
        # Check if resize is needed
        max_dimension = max(image.size)
        if max_dimension <= self.target_size[0]:
            return image

        # Calculate new size maintaining aspect ratio
        width, height = image.size
        aspect_ratio = width / height

        if width > height:
            new_width = self.target_size[0]
            new_height = int(new_width / aspect_ratio)
        else:
            new_height = self.target_size[1]
            new_width = int(new_height * aspect_ratio)

        # Resize with high-quality resampling
        resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        self.logger.debug(f"Resized from {image.size} to {resized.size}")
        return resized

    def image_to_bytes(self, image: Image.Image, format: str = 'PNG') -> bytes:
        """
        Convert PIL Image to bytes for API transmission

        Args:
            image: PIL Image
            format: Output format (PNG, JPEG)

        Returns:
            Image bytes
        """
        buffer = io.BytesIO()
        image.save(buffer, format=format)
        return buffer.getvalue()

    def get_image_info(self, image: Image.Image) -> Dict[str, Any]:
        """
        Get metadata about the image

        Returns:
            Dictionary with image information
        """
        return {
            'size': image.size,
            'width': image.width,
            'height': image.height,
            'mode': image.mode,
            'format': image.format,
            'aspect_ratio': image.width / image.height
        }
