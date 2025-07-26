import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from sklearn.metrics import precision_score, recall_score, f1_score
from scipy.stats import spearmanr, pearsonr
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from .config import EvaluationConfig
from .utils import debug_print

class ComprehensiveEvaluator:
    """Comprehensive evaluation framework for embedding models and retrieval systems"""
    
    def __init__(self, output_dir: str = "evaluation_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.results = {}
        
    def evaluate_retrieval_accuracy(self, 
                                   predicted_sections: List[Dict], 
                                   ground_truth_sections: List[Dict],
                                   k_values: List[int] = EvaluationConfig.RECALL_K_VALUES) -> Dict[str, float]:
        """Evaluate retrieval accuracy using multiple metrics"""
        
        results = {}
        
        # Calculate Mean Average Precision (MAP)
        results['map'] = self._calculate_map(predicted_sections, ground_truth_sections)
        
        # Calculate Recall@K and Precision@K for different K values
        for k in k_values:
            recall_k = self._calculate_recall_at_k(predicted_sections, ground_truth_sections, k)
            precision_k = self._calculate_precision_at_k(predicted_sections, ground_truth_sections, k)
            results[f'recall_at_{k}'] = recall_k
            results[f'precision_at_{k}'] = precision_k
        
        # Calculate F1 scores
        results['f1_macro'] = self._calculate_f1_score(predicted_sections, ground_truth_sections, average='macro')
        results['f1_micro'] = self._calculate_f1_score(predicted_sections, ground_truth_sections, average='micro')
        
        debug_print(f"Retrieval evaluation completed: MAP={results['map']:.4f}")
        return results
    
    def _calculate_map(self, predicted: List[Dict], ground_truth: List[Dict]) -> float:
        """Calculate Mean Average Precision (MAP)"""
        
        if not predicted or not ground_truth:
            return 0.0
        
        # Create relevance mapping
        gt_docs = {f"{item['document']}_{item['page']}": True for item in ground_truth}
        
        # Calculate Average Precision
        relevant_retrieved = 0
        average_precision = 0.0
        
        for i, pred_item in enumerate(predicted, 1):
            doc_page_key = f"{pred_item['document']}_{pred_item['page']}"
            
            if doc_page_key in gt_docs:
                relevant_retrieved += 1
                precision_at_i = relevant_retrieved / i
                average_precision += precision_at_i
        
        # Return MAP (in this case, just AP since we have one query)
        total_relevant = len(ground_truth)
        if total_relevant == 0:
            return 0.0
            
        return average_precision / total_relevant
    
    def _calculate_recall_at_k(self, predicted: List[Dict], ground_truth: List[Dict], k: int) -> float:
        """Calculate Recall@K"""
        
        if not predicted or not ground_truth or k <= 0:
            return 0.0
        
        # Take top-k predictions
        top_k_predicted = predicted[:k]
        
        # Create sets for comparison
        pred_docs = {f"{item['document']}_{item['page']}" for item in top_k_predicted}
        gt_docs = {f"{item['document']}_{item['page']}" for item in ground_truth}
        
        # Calculate recall
        intersection = pred_docs.intersection(gt_docs)
        recall = len(intersection) / len(gt_docs) if gt_docs else 0.0
        
        return recall
    
    def _calculate_precision_at_k(self, predicted: List[Dict], ground_truth: List[Dict], k: int) -> float:
        """Calculate Precision@K"""
        
        if not predicted or k <= 0:
            return 0.0
        
        # Take top-k predictions
        top_k_predicted = predicted[:k]
        
        # Create sets for comparison
        pred_docs = {f"{item['document']}_{item['page']}" for item in top_k_predicted}
        gt_docs = {f"{item['document']}_{item['page']}" for item in ground_truth}
        
        # Calculate precision
        intersection = pred_docs.intersection(gt_docs)
        precision = len(intersection) / len(pred_docs) if pred_docs else 0.0
        
        return precision
    
    def _calculate_f1_score(self, predicted: List[Dict], ground_truth: List[Dict], average: str = 'macro') -> float:
        """Calculate F1 score"""
        
        # Convert to binary classification problem
        all_docs = set()
        for item in predicted + ground_truth:
            all_docs.add(f"{item['document']}_{item['page']}")
        
        # Create binary vectors
        y_true = []
        y_pred = []
        
        gt_docs = {f"{item['document']}_{item['page']}" for item in ground_truth}
        pred_docs = {f"{item['document']}_{item['page']}" for item in predicted}
        
        for doc in all_docs:
            y_true.append(1 if doc in gt_docs else 0)
            y_pred.append(1 if doc in pred_docs else 0)
        
        return f1_score(y_true, y_pred, average=average, zero_division=0)
    
    def evaluate_semantic_similarity(self, 
                                   embeddings1: np.ndarray, 
                                   embeddings2: np.ndarray, 
                                   similarity_scores: np.ndarray) -> Dict[str, float]:
        """Evaluate semantic similarity against human judgments"""
        
        # Calculate cosine similarities
        cosine_sims = np.array([
            np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            for emb1, emb2 in zip(embeddings1, embeddings2)
        ])
        
        # Calculate correlations
        spearman_corr, spearman_p = spearmanr(cosine_sims, similarity_scores)
        pearson_corr, pearson_p = pearsonr(cosine_sims, similarity_scores)
        
        results = {
            'spearman_correlation': spearman_corr,
            'spearman_p_value': spearman_p,
            'pearson_correlation': pearson_corr,
            'pearson_p_value': pearson_p,
            'mean_cosine_similarity': np.mean(cosine_sims),
            'std_cosine_similarity': np.std(cosine_sims)
        }
        
        debug_print(f"Semantic similarity evaluation: Spearman={spearman_corr:.4f}")
        return results
    
    def benchmark_model_performance(self, model, test_data: Dict) -> Dict[str, Any]:
        """Comprehensive model benchmarking"""
        
        results = {
            'model_info': model.get_model_info(),
            'performance_metrics': {},
            'timing_metrics': {},
            'memory_metrics': {}
        }
        
        # Test embedding speed
        import time
        
        # Speed test
        test_texts = ["This is a test sentence."] * 1000
        start_time = time.time()
        embeddings = model.encode(test_texts)
        end_time = time.time()
        
        embedding_time = end_time - start_time
        texts_per_second = len(test_texts) / embedding_time
        
        results['timing_metrics'] = {
            'embedding_time_1k_texts': embedding_time,
            'texts_per_second': texts_per_second,
            'avg_time_per_text_ms': (embedding_time / len(test_texts)) * 1000
        }
        
        # Memory usage (basic estimation)
        import psutil
        import gc
        
        gc.collect()
        memory_before = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        # Generate some embeddings
        large_test = ["Test sentence for memory measurement."] * 5000
        large_embeddings = model.encode(large_test)
        
        memory_after = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        memory_usage = memory_after - memory_before
        
        results['memory_metrics'] = {
            'memory_usage_mb': memory_usage,
            'memory_per_embedding_kb': (memory_usage * 1024) / len(large_embeddings)
        }
        
        return results
    
    def save_results(self, results: Dict, filename: str):
        """Save evaluation results to file"""
        
        output_file = self.output_dir / f"{filename}.json"
        
        # Convert numpy types for JSON serialization
        json_results = self._convert_numpy_types(results)
        
        with open(output_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        debug_print(f"Results saved to {output_file}")
    
    def _convert_numpy_types(self, obj):
        """Convert numpy types to Python native types for JSON serialization"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        else:
            return obj
    
    def generate_evaluation_report(self, all_results: Dict):
        """Generate comprehensive evaluation report"""
        
        # Create summary statistics
        summary = {
            'total_evaluations': len(all_results),
            'average_map': np.mean([r.get('map', 0) for r in all_results.values()]),
            'average_recall_at_5': np.mean([r.get('recall_at_5', 0) for r in all_results.values()]),
            'best_performing_collection': max(all_results.keys(), 
                                            key=lambda k: all_results[k].get('map', 0))
        }
        
        # Save detailed report
        report = {
            'summary': summary,
            'detailed_results': all_results,
            'evaluation_config': {
                'metrics_used': EvaluationConfig.METRICS,
                'recall_k_values': EvaluationConfig.RECALL_K_VALUES
            }
        }
        
        self.save_results(report, 'comprehensive_evaluation_report')
        
        # Generate visualizations
        self._create_evaluation_plots(all_results)
        
        return report
    
    def _create_evaluation_plots(self, results: Dict):
        """Create evaluation visualizations"""
        
        # Extract metrics for plotting
        collections = list(results.keys())
        map_scores = [results[col].get('map', 0) for col in collections]
        recall_5_scores = [results[col].get('recall_at_5', 0) for col in collections]
        
        # Create plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # MAP scores
        ax1.bar(collections, map_scores)
        ax1.set_title('Mean Average Precision by Collection')
        ax1.set_ylabel('MAP Score')
        ax1.tick_params(axis='x', rotation=45)
        
        # Recall@5 scores
        ax2.bar(collections, recall_5_scores)
        ax2.set_title('Recall@5 by Collection')
        ax2.set_ylabel('Recall@5 Score')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'evaluation_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        debug_print("Evaluation plots saved")
 