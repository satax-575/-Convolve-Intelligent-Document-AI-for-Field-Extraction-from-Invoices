#!/usr/bin/env python3
"""
Main Executable: Document AI Field Extraction Pipeline
Follows 6-stage architecture as per problem statement
"""
import os
import sys
import json
import time
import glob
from pathlib import Path

# Import custom modules
from utils.ingestion import DocumentIngestion
from utils.vision import VisionModule
from utils.ocr import OCRModule
from utils.extraction import FieldExtractor
from utils.postprocessing import PostProcessor

class DocumentAIPipeline:
    def __init__(self):
        # Initialize all modules
        self.ingestion = DocumentIngestion(dpi=200)
        self.vision = VisionModule()
        self.ocr = OCRModule(languages=['en', 'hi'])
        self.extractor = FieldExtractor()
        self.postproc = PostProcessor()

    def process_document(self, file_path):
        """
        Main pipeline: Processes one document
        Returns: JSON output matching required format
        """
        doc_id = Path(file_path).stem
        start_time = time.time()

        try:
            # Stage 1: Ingestion
            images = self.ingestion.load_document(file_path)
            image = images[0]  # Process first page
            image = self.ingestion.preprocess(image)

            # Stage 2: Parallel extraction
            # 2a. Visual
            visual_data = self.vision.detect_visual_elements(image)

            # 2b. Textual
            ocr_lines, full_text = self.ocr.extract_text(image)

            # Stage 3 & 4: Field detection and reasoning
            fields = self.extractor.extract_all_fields(ocr_lines, full_text)

            # Stage 5: Post-processing
            avg_ocr_conf = sum([l['confidence'] for l in ocr_lines]) / len(ocr_lines) if ocr_lines else 0
            confidence = self.postproc.validate_and_score(fields, avg_ocr_conf, visual_data)
            fields = self.postproc.normalize_output(fields)

            # Stage 6: Output generation
            processing_time = time.time() - start_time
            cost_estimate = (0.35 / 3600) * processing_time  # T4 GPU hourly cost

            output = {
                "doc_id": doc_id,
                "fields": {
                    "dealer_name": fields['dealer_name'],
                    "model_name": fields['model_name'],
                    "horse_power": fields['horse_power'],
                    "asset_cost": fields['asset_cost'],
                    "signature": visual_data['signature'],
                    "stamp": visual_data['stamp']
                },
                "confidence": confidence,
                "processing_time_sec": round(processing_time, 2),
                "cost_estimate_usd": round(cost_estimate, 5)
            }

            return output

        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            return None

    def batch_process(self, input_folder, output_file='result.json'):
        """Process all documents in folder"""
        # Find all documents
        files = glob.glob(os.path.join(input_folder, '*.pdf')) +                 glob.glob(os.path.join(input_folder, '*.jpg')) +                 glob.glob(os.path.join(input_folder, '*.png'))

        print(f"Found {len(files)} documents to process")

        results = []
        for idx, file_path in enumerate(files, 1):
            print(f"[{idx}/{len(files)}] Processing: {Path(file_path).name}")
            result = self.process_document(file_path)
            if result:
                results.append(result)

        # Save output
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"
Processed {len(results)} documents")
        print(f"Output saved to: {output_file}")

        return results

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python executable.py <input_folder>")
        sys.exit(1)

    input_folder = sys.argv[1]
    pipeline = DocumentAIPipeline()
    pipeline.batch_process(input_folder)
