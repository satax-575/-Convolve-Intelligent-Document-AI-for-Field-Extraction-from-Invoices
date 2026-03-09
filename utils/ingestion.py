"""
Stage 1: Document Ingestion and Interpretation
Converts PDFs/images into processable format
"""
import os
import numpy as np
from PIL import Image
from pdf2image import convert_from_path
import cv2

class DocumentIngestion:
    def __init__(self, dpi=200):
        self.dpi = dpi

    def load_document(self, file_path):
        """
        Convert input document to image array.
        Handles: PDF, JPG, PNG
        Returns: list of numpy arrays (one per page)
        """
        ext = os.path.splitext(file_path)[1].lower()

        if ext == '.pdf':
            return self._load_pdf(file_path)
        elif ext in ['.jpg', '.jpeg', '.png']:
            return self._load_image(file_path)
        else:
            raise ValueError(f"Unsupported format: {ext}")

    def _load_pdf(self, pdf_path):
        """Convert PDF to images"""
        images = convert_from_path(pdf_path, dpi=self.dpi)
        # Convert PIL to numpy
        return [np.array(img) for img in images]

    def _load_image(self, img_path):
        """Load image file"""
        img = Image.open(img_path).convert('RGB')
        return [np.array(img)]

    def preprocess(self, image):
        """
        Quality enhancement for better OCR
        - Denoise
        - Adaptive thresholding (if needed)
        """
        # Convert to grayscale for quality check
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Check if image is too dark/bright
        mean_intensity = np.mean(gray)

        if mean_intensity < 80:  # Too dark
            image = cv2.convertScaleAbs(image, alpha=1.5, beta=30)
        elif mean_intensity > 200:  # Too bright
            image = cv2.convertScaleAbs(image, alpha=0.8, beta=-20)

        # Denoise
        image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

        return image
