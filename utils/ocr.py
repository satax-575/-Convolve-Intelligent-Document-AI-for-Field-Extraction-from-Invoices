"""
Stage 2: Textual Understanding - OCR Extraction
Handles multilingual text extraction
"""
import easyocr
import numpy as np

class OCRModule:
    def __init__(self, languages=['en', 'hi']):
        self.reader = easyocr.Reader(languages, gpu=True, verbose=False)

    def extract_text(self, image):
        """
        Extract text with bounding boxes and confidence
        Returns: list of {text, bbox, conf}
        """
        results = self.reader.readtext(image, detail=1, paragraph=False)

        structured_output = []
        for (bbox, text, conf) in results:
            structured_output.append({
                'text': text,
                'bbox': bbox,
                'confidence': float(conf)
            })

        # Also return full text for easier processing
        full_text = ' '.join([item['text'] for item in structured_output])

        return structured_output, full_text
