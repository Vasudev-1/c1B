import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sentence_transformers import SentenceTransformer, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
import numpy as np
from typing import List, Tuple, Dict
import logging

from .config import OptimizationConfig
from .utils import debug_print

class DistillationDataset(Dataset):
    """Dataset for knowledge distillation"""
    
    def __init__(self, texts: List[str], teacher_embeddings: np.ndarray):
        self.texts = texts
        self.teacher_embeddings = teacher_embeddings
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return {
            'text': self.texts[idx],
            'teacher_embedding': torch.FloatTensor(self.teacher_embeddings[idx])
        }

class KnowledgeDistillationLoss(nn.Module):
    """Custom loss for knowledge distillation"""
    
    def __init__(self, temperature: float = 4.0, alpha: float = 0.7):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.mse_loss = nn.MSELoss()
        self.cosine_loss = nn.CosineEmbeddingLoss()
    
    def forward(self, student_embeddings: torch.Tensor, 
                teacher_embeddings: torch.Tensor) -> torch.Tensor:
        
        # MSE loss for embedding alignment
        mse_loss = self.mse_loss(student_embeddings, teacher_embeddings)
        
        # Cosine similarity loss
        target = torch.ones(student_embeddings.size(0)).to(student_embeddings.device)
        cosine_loss = self.cosine_loss(student_embeddings, teacher_embeddings, target)
        
        # Combined loss
        total_loss = self.alpha * mse_loss + (1 - self.alpha) * cosine_loss
        
        return total_loss

class EmbeddingModelTrainer:
    """Advanced trainer for embedding models with distillation and fine-tuning"""
    
    def __init__(self, 
                 student_model_name: str,
                 teacher_model_name: str = OptimizationConfig.TEACHER_MODEL,
                 output_dir: str = "models/fine_tuned"):
        
        self.student_model_name = student_model_name
        self.teacher_model_name = teacher_model_name
        self.output_dir = output_dir
        
        # Load models
        self.student_model = SentenceTransformer(student_model_name)
        self.teacher_model = SentenceTransformer(teacher_model_name)
        
        debug_print(f"Loaded student: {student_model_name}")
        debug_print(f"Loaded teacher: {teacher_model_name}")
    
    def knowledge_distillation(self, 
                             training_texts: List[str],
                             validation_texts: List[str] = None,
                             epochs: int = OptimizationConfig.EPOCHS,
                             batch_size: int = OptimizationConfig.BATCH_SIZE) -> SentenceTransformer:
        """Perform knowledge distillation from teacher to student"""
        
        debug_print("Starting knowledge distillation...")
        
        # Generate teacher embeddings
        debug_print("Generating teacher embeddings...")
        teacher_embeddings = self.teacher_model.encode(training_texts, 
                                                     convert_to_numpy=True,
                                                     show_progress_bar=True)
        
        # Create dataset
        train_dataset = DistillationDataset(training_texts, teacher_embeddings)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Define loss function
        distillation_loss = KnowledgeDistillationLoss(
            temperature=OptimizationConfig.DISTILLATION_TEMPERATURE,
            alpha=OptimizationConfig.DISTILLATION_ALPHA
        )
        
        # Training loop
        self.student_model.to('cuda' if torch.cuda.is_available() else 'cpu')
        optimizer = torch.optim.AdamW(self.student_model.parameters(), 
                                    lr=OptimizationConfig.LEARNING_RATE)
        
        for epoch in range(epochs):
            total_loss = 0
            num_batches = 0
            
            for batch in train_dataloader:
                optimizer.zero_grad()
                
                # Get student embeddings
                student_embeddings = self.student_model.encode(batch['text'], 
                                                             convert_to_tensor=True)
                teacher_embeddings = batch['teacher_embedding'].to(student_embeddings.device)
                
                # Calculate loss
                loss = distillation_loss(student_embeddings, teacher_embeddings)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches
            debug_print(f"Epoch {epoch + 1}/{epochs}: Average Loss = {avg_loss:.4f}")
        
        # Save distilled model
        distilled_model_path = f"{self.output_dir}/distilled_{self.student_model_name.replace('/', '_')}"
        self.student_model.save(distilled_model_path)
        
        debug_print(f"Distillation completed. Model saved to {distilled_model_path}")
        return self.student_model
    
    def fine_tune_on_domain_data(self, 
                                training_pairs: List[Tuple[str, str, float]],
                                validation_pairs: List[Tuple[str, str, float]] = None,
                                epochs: int = OptimizationConfig.EPOCHS) -> SentenceTransformer:
        """Fine-tune model on domain-specific data"""
        
        debug_print("Starting domain-specific fine-tuning...")
        
        # Prepare training data
        train_examples = []
        for text1, text2, score in training_pairs:
            train_examples.append({
                'texts': [text1, text2],
                'label': score
            })
        
        # Create data loader
        train_dataloader = DataLoader(train_examples, batch_size=16, shuffle=True)
        
        # Define loss function (CosineSimilarityLoss for similarity tasks)
        train_loss = losses.CosineSimilarityLoss(self.student_model)
        
        # Set up evaluation
        evaluator = None
        if validation_pairs:
            val_examples = []
            for text1, text2, score in validation_pairs:
                val_examples.append({
                    'sentence1': text1,
                    'sentence2': text2,
                    'score': score
                })
            evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
                val_examples, write_csv=True
            )
        
        # Fine-tune
        self.student_model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=epochs,
            evaluator=evaluator,
            evaluation_steps=1000,
            warmup_steps=100,
            output_path=f"{self.output_dir}/fine_tuned_{self.student_model_name.replace('/', '_')}"
        )
        
        debug_print("Fine-tuning completed")
        return self.student_model
    
    def create_training_data_from_hackathon(self, collections_data: Dict) -> List[str]:
        """Create training data from hackathon challenge data"""
        
        training_texts = []
        
        for collection_name, collection_data in collections_data.items():
            # Extract texts from chunks
            for doc_data in collection_data['documents'].values():
                training_texts.extend(doc_data['chunks'])
            
            # Add persona and job descriptions
            training_texts.append(collection_data['persona'])
            training_texts.append(collection_data['job_to_be_done'])
        
        debug_print(f"Created training dataset with {len(training_texts)} texts")
        return training_texts
