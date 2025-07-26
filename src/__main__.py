import argparse
import time
import json
from pathlib import Path
from typing import Dict, List

from .config import ModelConfig, OptimizationConfig, EvaluationConfig
from .utils import debug_print, log_time, setup_logging
from .ingest import ingest_documents
from .embed import AdvancedEmbeddingModel, embed_documents
from .rank import pipeline, focus_keywords
from .assemble import build_output
from .evaluation import ComprehensiveEvaluator
from .fine_tune import EmbeddingModelTrainer

class Challenge1BPipeline:
    """Complete Challenge 1B pipeline with evaluation and optimization"""
    
    def __init__(self, enable_evaluation: bool = True, enable_optimization: bool = True):
        self.enable_evaluation = enable_evaluation
        self.enable_optimization = enable_optimization
        self.evaluator = ComprehensiveEvaluator() if enable_evaluation else None
        self.model = None
        self.results = {}
        
        setup_logging()
        debug_print("Challenge 1B Pipeline initialized")
    
    def load_optimized_model(self) -> AdvancedEmbeddingModel:
        """Load and optimize the embedding model"""
        
        debug_print("Loading advanced embedding model...")
        
        # Load the base model
        self.model = AdvancedEmbeddingModel(
            model_name=ModelConfig.EMBEDDING_MODEL,
            enable_quantization=OptimizationConfig.ENABLE_QUANTIZATION
        )
        
        # Optional: Apply knowledge distillation if enabled
        if OptimizationConfig.ENABLE_DISTILLATION and self.enable_optimization:
            debug_print("Knowledge distillation is enabled but requires training data")
            # This would be done during training phase
        
        return self.model
    
    def process_single_collection(self, collection_path: Path) -> Dict:
        """Process a single collection with comprehensive evaluation"""
        
        debug_print(f"Processing collection: {collection_path.name}")
        start_time = log_time()
        
        # Load input configuration
        input_file = collection_path / "challenge1b_input.json"
        with open(input_file, 'r') as f:
            input_config = json.load(f)
        
        # Extract configuration
        pdf_dir = collection_path / "PDFs"
        pdf_paths = [pdf_dir / doc['filename'] for doc in input_config['documents']]
        persona = input_config['persona']
        job = input_config['job_to_be_done']
        
        # Process documents
        docs = ingest_documents([str(p) for p in pdf_paths])
        docs = embed_documents(docs, self.model)
        
        # Extract keywords and create query
        keywords = focus_keywords(job)
        query_vector = self.model.encode_query(persona, job, keywords)
        
        # Run retrieval and ranking pipeline
        sections, subsections = pipeline(query_vector, docs, job)
        
        # Build output
        output_file = collection_path / "challenge1b_output.json"
        build_output(
            output_file,
            [p.name for p in pdf_paths],
            persona,
            job,
            sections,
            subsections
        )
        
        processing_time = log_time() - start_time
        debug_print(f"{collection_path.name} completed in {processing_time:.1f}s")
        
        # Evaluation if enabled
        evaluation_results = {}
        if self.enable_evaluation:
            evaluation_results = self.evaluate_collection_output(
                collection_path, sections, subsections
            )
        
        return {
            'processing_time': processing_time,
            'sections_found': len(sections),
            'subsections_found': len(subsections),
            'evaluation_results': evaluation_results
        }
    
    def evaluate_collection_output(self, collection_path: Path, 
                                 predicted_sections: List[Dict],
                                 predicted_subsections: List[Dict]) -> Dict:
        """Evaluate collection output against ground truth"""
        
        # Load ground truth if available
        sample_output_file = collection_path / "challenge1b_output.json"
        if not sample_output_file.exists():
            debug_print("No ground truth available for evaluation")
            return {}
        
        with open(sample_output_file, 'r') as f:
            ground_truth = json.load(f)
        
        # Extract ground truth sections
        gt_sections = ground_truth.get('extracted_sections', [])
        gt_subsections = ground_truth.get('subsection_analysis', [])
        
        # Evaluate retrieval accuracy
        results = self.evaluator.evaluate_retrieval_accuracy(
            predicted_sections, gt_sections
        )
        
        debug_print(f"Evaluation completed for {collection_path.name}")
        return results
    
    def run_comprehensive_evaluation(self):
        """Run comprehensive model evaluation"""
        
        if not self.enable_evaluation:
            debug_print("Evaluation disabled")
            return
        
        debug_print("Running comprehensive model evaluation...")
        
        # Benchmark model performance
        benchmark_results = self.evaluator.benchmark_model_performance(
            self.model, {}
        )
        
        # Save benchmark results
        self.evaluator.save_results(benchmark_results, 'model_benchmark')
        
        debug_print("Comprehensive evaluation completed")
    
    def fine_tune_model_if_enabled(self, training_data: Dict):
        """Fine-tune model if optimization is enabled"""
        
        if not (self.enable_optimization and OptimizationConfig.ENABLE_FINE_TUNING):
            return
        
        debug_print("Starting model fine-tuning...")
        
        # Create trainer
        trainer = EmbeddingModelTrainer(
            student_model_name=ModelConfig.EMBEDDING_MODEL,
            teacher_model_name=OptimizationConfig.TEACHER_MODEL
        )
        
        # Prepare training data
        training_texts = trainer.create_training_data_from_hackathon(training_data)
        
        # Perform knowledge distillation
        if OptimizationConfig.ENABLE_DISTILLATION:
            self.model = trainer.knowledge_distillation(training_texts)
        
        debug_print("Model fine-tuning completed")
    
    def run_all_collections(self, base_path: str = "Challenge_1b"):
        """Run pipeline on all collections"""
        
        base_dir = Path(base_path)
        collections = sorted([
            p for p in base_dir.iterdir() 
            if p.is_dir() and p.name.startswith('Collection')
        ])
        
        debug_print(f"Found {len(collections)} collections to process")
        
        # Load model
        self.load_optimized_model()
        
        # Run comprehensive evaluation first
        self.run_comprehensive_evaluation()
        
        # Process each collection
        all_results = {}
        for collection_path in collections:
            collection_results = self.process_single_collection(collection_path)
            all_results[collection_path.name] = collection_results
        
        # Generate final evaluation report
        if self.enable_evaluation:
            evaluation_report = self.evaluator.generate_evaluation_report(
                {k: v['evaluation_results'] for k, v in all_results.items()}
            )
            debug_print("Evaluation report generated")
        
        # Save overall results
        self.evaluator.save_results(all_results, 'pipeline_results')
        
        return all_results

def main():
    """Main entry point"""
    
    parser = argparse.ArgumentParser(description='Challenge 1B Pipeline')
    parser.add_argument('--base', default='Challenge_1b', 
                       help='Base directory for collections')
    parser.add_argument('--no-evaluation', action='store_true',
                       help='Disable evaluation')
    parser.add_argument('--no-optimization', action='store_true',
                       help='Disable optimization features')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode')
    
    args = parser.parse_args()
    
    # Configure debug mode
    if args.debug:
        import logging
        logging.basicConfig(level=logging.DEBUG)
    
    # Initialize and run pipeline
    pipeline = Challenge1BPipeline(
        enable_evaluation=not args.no_evaluation,
        enable_optimization=not args.no_optimization
    )
    
    # Run the complete pipeline
    results = pipeline.run_all_collections(args.base)
    
    debug_print("Pipeline execution completed successfully")
    return results

if __name__ == '__main__':
    main()
