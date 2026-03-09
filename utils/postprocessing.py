"""
Stage 5: Post-Processing and Quality Assurance
Validation, confidence scoring, normalization
"""
import re

class PostProcessor:
    def __init__(self):
        pass

    def validate_and_score(self, extracted_fields, ocr_confidence, visual_data):
        """
        Apply validation rules and compute overall confidence
        """
        # Individual field validation
        validations = {
            'dealer_name': self._validate_text(extracted_fields.get('dealer_name')),
            'model_name': self._validate_text(extracted_fields.get('model_name')),
            'horse_power': self._validate_numeric(extracted_fields.get('horse_power'), 20, 200),
            'asset_cost': self._validate_numeric(extracted_fields.get('asset_cost'), 100000, 10000000)
        }

        # Overall confidence: weighted average
        field_conf = sum(validations.values()) / len(validations)
        overall_conf = (field_conf * 0.6) + (ocr_confidence * 0.4)

        return round(overall_conf, 3)

    def _validate_text(self, value):
        """Check if text field is valid"""
        if value and len(value) > 2:
            return 1.0
        elif value:
            return 0.5
        return 0.0

    def _validate_numeric(self, value, min_val, max_val):
        """Check if numeric field is in expected range"""
        if value and min_val <= value <= max_val:
            return 1.0
        elif value:
            return 0.5
        return 0.0

    def normalize_output(self, data):
        """
        Final output normalization
        - Remove None values (replace with empty)
        - Ensure correct data types
        """
        if data.get('dealer_name') is None:
            data['dealer_name'] = ""
        if data.get('model_name') is None:
            data['model_name'] = ""
        if data.get('horse_power') is None:
            data['horse_power'] = 0
        if data.get('asset_cost') is None:
            data['asset_cost'] = 0

        return data
