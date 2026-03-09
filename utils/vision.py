"""
Stage 2: Visual Understanding - Signature & Stamp Detection
Uses YOLO for object detection
"""
from ultralytics import YOLO
import numpy as np

class VisionModule:
    def __init__(self, model_path='yolov8n.pt', conf_threshold=0.25):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold

    def detect_visual_elements(self, image):
        """
        Detect stamps and signatures in document
        Returns: dict with presence flags and bounding boxes
        """
        results = self.model.predict(image, conf=self.conf_threshold, verbose=False)[0]

        output = {
            "signature": {"present": False, "bbox": []},
            "stamp": {"present": False, "bbox": []}
        }

        # If using a custom trained model, map class IDs
        # For demo: using heuristic on generic YOLO detections
        # In production: replace with trained model class mappings

        if len(results.boxes) > 0:
            for box in results.boxes:
                # Get bbox coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])

                # Heuristic: Assume stamps are larger, signatures smaller
                bbox_area = (x2 - x1) * (y2 - y1)
                image_area = image.shape[0] * image.shape[1]
                relative_size = bbox_area / image_area

                if relative_size > 0.01 and relative_size < 0.05:
                    # Likely a stamp
                    if not output["stamp"]["present"]:
                        output["stamp"]["present"] = True
                        output["stamp"]["bbox"] = [int(x1), int(y1), int(x2), int(y2)]
                elif relative_size > 0.005 and relative_size < 0.02:
                    # Likely a signature
                    if not output["signature"]["present"]:
                        output["signature"]["present"] = True
                        output["signature"]["bbox"] = [int(x1), int(y1), int(x2), int(y2)]

        return output
