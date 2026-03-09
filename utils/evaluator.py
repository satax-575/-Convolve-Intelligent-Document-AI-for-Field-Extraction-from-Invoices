"""
Evaluation Module: Calculate DLA and Secondary Metrics
Implements exact evaluation rules from problem statement
"""
import numpy as np
from rapidfuzz import fuzz

class Evaluator:
    def __init__(self):
        self.fuzzy_threshold = 90  # 90% for dealer/model
        self.numeric_tolerance = 0.05  # ±5%
        self.iou_threshold = 0.5  # IoU ≥ 0.5

    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union for bounding boxes"""
        if not box1 or not box2:
            return 0.0

        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2

        # Intersection
        xi_min = max(x1_min, x2_min)
        yi_min = max(y1_min, y2_min)
        xi_max = min(x1_max, x2_max)
        yi_max = min(y1_max, y2_max)

        if xi_max < xi_min or yi_max < yi_min:
            return 0.0

        intersection = (xi_max - xi_min) * (yi_max - yi_min)

        # Union
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union = box1_area + box2_area - intersection

        return intersection / union if union > 0 else 0.0

    def check_dealer_match(self, predicted, ground_truth):
        """Fuzzy match for dealer name (≥90%)"""
        if not predicted or not ground_truth:
            return False
        score = fuzz.token_set_ratio(predicted, ground_truth)
        return score >= self.fuzzy_threshold

    def check_model_match(self, predicted, ground_truth):
        """Exact or fuzzy match for model (≥90%)"""
        if not predicted or not ground_truth:
            return False
        # Try exact first
        if predicted.lower() == ground_truth.lower():
            return True
        # Fuzzy fallback
        score = fuzz.ratio(predicted, ground_truth)
        return score >= self.fuzzy_threshold

    def check_numeric_match(self, predicted, ground_truth):
        """Numeric equality within ±5% tolerance"""
        if predicted is None or ground_truth is None:
            return False
        if ground_truth == 0:
            return predicted == 0
        tolerance = abs(ground_truth * self.numeric_tolerance)
        return abs(predicted - ground_truth) <= tolerance

    def check_visual_match(self, predicted, ground_truth):
        """Check presence + IoU ≥ 0.5"""
        # Check presence
        if predicted['present'] != ground_truth['present']:
            return False
        # If both absent, it's a match
        if not predicted['present']:
            return True
        # Check IoU
        iou = self.calculate_iou(predicted['bbox'], ground_truth['bbox'])
        return iou >= self.iou_threshold

    def evaluate_document(self, predicted, ground_truth):
        """
        Evaluate single document against ground truth
        Returns: dict with per-field results
        """
        results = {
            'dealer_name': self.check_dealer_match(
                predicted['fields']['dealer_name'],
                ground_truth['dealer_name']
            ),
            'model_name': self.check_model_match(
                predicted['fields']['model_name'],
                ground_truth['model_name']
            ),
            'horse_power': self.check_numeric_match(
                predicted['fields']['horse_power'],
                ground_truth['horse_power']
            ),
            'asset_cost': self.check_numeric_match(
                predicted['fields']['asset_cost'],
                ground_truth['asset_cost']
            ),
            'signature': self.check_visual_match(
                predicted['fields']['signature'],
                ground_truth['signature']
            ),
            'stamp': self.check_visual_match(
                predicted['fields']['stamp'],
                ground_truth['stamp']
            )
        }

        # Document-level accuracy: ALL fields must be correct
        results['document_correct'] = all(results.values())

        return results

    def calculate_dla(self, predictions, ground_truths):
        """
        Calculate Document-Level Accuracy (DLA)
        Target: ≥95%
        """
        total_docs = len(predictions)
        correct_docs = 0
        field_accuracies = {
            'dealer_name': 0, 'model_name': 0, 'horse_power': 0,
            'asset_cost': 0, 'signature': 0, 'stamp': 0
        }

        detailed_results = []

        for pred, gt in zip(predictions, ground_truths):
            result = self.evaluate_document(pred, gt)
            detailed_results.append(result)

            if result['document_correct']:
                correct_docs += 1

            # Track per-field accuracy
            for field in field_accuracies:
                if result[field]:
                    field_accuracies[field] += 1

        dla = (correct_docs / total_docs * 100) if total_docs > 0 else 0

        # Per-field accuracy
        for field in field_accuracies:
            field_accuracies[field] = (field_accuracies[field] / total_docs * 100) if total_docs > 0 else 0

        return {
            'DLA': round(dla, 2),
            'correct_documents': correct_docs,
            'total_documents': total_docs,
            'field_accuracies': field_accuracies,
            'detailed_results': detailed_results
        }
