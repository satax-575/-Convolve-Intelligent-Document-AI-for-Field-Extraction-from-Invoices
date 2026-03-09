"""
Stages 3 & 4: Field Detection and Semantic Reasoning
Rule-based + Fuzzy matching extraction
"""
import re
from rapidfuzz import process, fuzz

class FieldExtractor:
    def __init__(self, master_dealers=None, master_models=None):
        # Master lists for fuzzy matching
        self.master_dealers = master_dealers or [
            "Sai Tractors", "Kisan Motors", "Mahindra Agency",
            "Gujarat Agro", "ABC Tractors Pvt Ltd"
        ]
        self.master_models = master_models or [
            "Mahindra 575 DI", "Swaraj 744 FE", "John Deere 5310",
            "Sonalika DI 750", "Eicher 380"
        ]

    def extract_all_fields(self, ocr_lines, full_text):
        """Extract all 4 text fields"""
        return {
            'dealer_name': self._extract_dealer(ocr_lines, full_text),
            'model_name': self._extract_model(ocr_lines, full_text),
            'horse_power': self._extract_hp(full_text),
            'asset_cost': self._extract_cost(ocr_lines, full_text)
        }

    def _extract_dealer(self, ocr_lines, full_text):
        """Extract and fuzzy match dealer name"""
        candidate = None

        # Strategy 1: Look for "Dealer" keyword
        for item in ocr_lines:
            text = item['text']
            if 'dealer' in text.lower():
                candidate = re.sub(r'dealer\s*[:\-]?', '', text, flags=re.IGNORECASE).strip()
                break

        # Strategy 2: First line (common in invoices)
        if not candidate and ocr_lines:
            candidate = ocr_lines[0]['text']

        # Fuzzy match against master list
        if candidate:
            match = process.extractOne(
                candidate,
                self.master_dealers,
                scorer=fuzz.token_set_ratio
            )
            if match and match[1] > 80:  # 80% threshold
                return match[0]
            return candidate

        return None

    def _extract_model(self, ocr_lines, full_text):
        """Exact match model name from master list"""
        # Try exact match first
        for model in self.master_models:
            if model.lower() in full_text.lower():
                return model

        # Fallback: look for "Model" keyword
        for item in ocr_lines:
            if 'model' in item['text'].lower():
                return re.sub(r'model\s*[:\-]?', '', item['text'], flags=re.IGNORECASE).strip()

        return None

    def _extract_hp(self, full_text):
        """Extract horse power (numeric)"""
        pattern = r'(\d{2,3})\s*(?:HP|H\.P\.|Hp|hp)'
        match = re.search(pattern, full_text)
        return int(match.group(1)) if match else None

    def _extract_cost(self, ocr_lines, full_text):
        """Extract asset cost (numeric)"""
        def clean_num(s):
            s = re.sub(r'[^\d.]', '', s)
            try: return float(s)
            except: return 0

        candidates = []

        # Look near keywords
        for item in ocr_lines:
            text = item['text']
            if re.search(r'(Total|Grand|Price|Amount|Only)', text, re.IGNORECASE):
                val = clean_num(text)
                if val > 100000:  # Sanity check
                    candidates.append(val)

        # Fallback: largest number > 200k
        if not candidates:
            all_nums = [clean_num(item['text']) for item in ocr_lines]
            candidates = [n for n in all_nums if 200000 < n < 10000000]

        return int(max(candidates)) if candidates else None
