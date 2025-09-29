import pandas as pd
from scipy import sparse
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler
import warnings
import logging
import time

class TextClassification():
    """
    Optimized text classification class for various ML algorithms.
    Supports k-Nearest Neighbors (KNN) and Multi-Layer Perceptron (MLP) with performance optimizations.
    """
    
    def __init__(self, model_type='knn', **kwargs):
        """
        Initialize the classifier with optimized parameters.
        
        Args:
            model_type (str): Type of model ('knn' or 'mlp')
            **kwargs: Additional parameters for the respective sklearn model
        """
        self.model_type = model_type
        self.is_trained = False
        self.scaler = None
        self.training_time = None
        
        if model_type == 'knn':
            # KNN works well with sparse matrices and doesn't need scaling
            default_knn_params = {
                'n_neighbors': 5,
                'algorithm': 'auto',  # Let sklearn choose the best algorithm
                'n_jobs': -1  # Use all available CPU cores
            }
            default_knn_params.update(kwargs)
            self.model = KNeighborsClassifier(**default_knn_params)
            
        elif model_type == 'mlp':
            # Optimized MLP parameters for text classification
            default_mlp_params = {
                'hidden_layer_sizes': (100, 50),  # Smaller, more efficient architecture
                'max_iter': 500,  # Higher limit but with early stopping
                'early_stopping': True,  # Stop when validation score stops improving
                'validation_fraction': 0.1,  # Use 10% for validation during training
                'n_iter_no_change': 10,  # Stop if no improvement for 10 iterations
                'learning_rate_init': 0.01,  # Better learning rate for text data
                'solver': 'adam',  # Good for large datasets
                'alpha': 0.001,  # L2 regularization
                'random_state': 42,  # For reproducibility
                'verbose': False  # Reduce output noise
            }
            default_mlp_params.update(kwargs)
            self.model = MLPClassifier(**default_mlp_params)
            
            # MLP needs feature scaling, especially for sparse matrices
            self.scaler = StandardScaler()
            
        else:
            raise ValueError(f"Unknown model type: {model_type}. Supported: 'knn', 'mlp'")
        
        logging.info(f"Initialized {model_type.upper()} classifier with optimized parameters")
    
    def _handle_sparse_matrix(self, X):
        """
        Safely handle sparse matrices to avoid length ambiguity errors.
        
        Args:
            X: Input matrix (sparse or dense)
            
        Returns:
            Processed matrix and information about sparsity
        """
        is_sparse = sparse.issparse(X)
        
        if is_sparse:
            # Ensure proper sparse matrix format
            if not sparse.isspmatrix_csr(X):
                X = X.tocsr()
            
            # Get dimensions safely
            n_samples, n_features = X.shape
            logging.info(f"Processing sparse matrix: {n_samples} samples, {n_features} features")
            
        else:
            # Handle dense matrices
            if len(X.shape) != 2:
                X = np.atleast_2d(X)
            n_samples, n_features = X.shape
            logging.info(f"Processing dense matrix: {n_samples} samples, {n_features} features")
        
        return X, is_sparse
    
    def fit(self, X_train, y_train):
        """
        Train the model with optimized preprocessing.
        
        Args:
            X_train: Training feature matrix
            y_train: Training labels
        """
        start_time = time.time()
        
        try:
            # Handle matrix format and get info
            X_train, is_sparse = self._handle_sparse_matrix(X_train)
            
            # Validate labels
            if sparse.issparse(y_train):
                y_train = y_train.toarray()
            y_train = np.asarray(y_train).ravel()
            
            logging.info(f"Training {self.model_type.upper()} with {X_train.shape[0]} samples")
            
            # Apply scaling for MLP
            if self.model_type == 'mlp' and self.scaler is not None:
                logging.info("Applying feature scaling for MLP...")

                # if sparce matrix not center data with mean = false, else center datasets
                if is_sparse:
                    self.scaler.with_mean = False
                else:
                    self.scaler.with_mean = True

                X_train = self.scaler.fit_transform(X_train)
            
            # Train the model with warnings suppressed for cleaner output
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                self.model.fit(X_train, y_train)
            
            self.training_time = time.time() - start_time
            self.is_trained = True
            
            # Log training results
            if self.model_type == 'mlp' and hasattr(self.model, 'n_iter_'):
                logging.info(f"MLP converged after {self.model.n_iter_} iterations in {self.training_time:.2f} seconds")
            else:
                logging.info(f"{self.model_type.upper()} trained successfully in {self.training_time:.2f} seconds")
                
        except Exception as e:
            logging.error(f"Error training {self.model_type}: {str(e)}")
            raise
    
    def predict(self, X_test):
        """
        Make predictions with proper preprocessing.
        
        Args:
            X_test: Test feature matrix
            
        Returns:
            Prediction array
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions.")
        
        try:
            # Handle matrix format
            X_test, _ = self._handle_sparse_matrix(X_test)
            
            # Apply same scaling as training for MLP
            if self.model_type == 'mlp' and self.scaler is not None:
                X_test = self.scaler.transform(X_test)
            
            # Make predictions
            predictions = self.model.predict(X_test)
            return predictions
            
        except Exception as e:
            logging.error(f"Error making predictions with {self.model_type}: {str(e)}")
            raise
    
    def evaluate(self, X_test, y_test, verbose=True):
        """
        Evaluate the model with comprehensive metrics.
        
        Args:
            X_test: Test feature matrix
            y_test: Test labels
            verbose: Whether to print detailed results
            
        Returns:
            tuple: (accuracy, f1_score)
        """
        try:
            start_time = time.time()
            y_pred = self.predict(X_test)
            prediction_time = time.time() - start_time
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            if verbose:
                print(f"\n=== {self.model_type.upper()} Evaluation ===")
                print(f"Training Time: {self.training_time:.2f} seconds")
                print(f"Prediction Time: {prediction_time:.2f} seconds")
                print(f"Accuracy: {accuracy:.4f}")
                print(f"F1-Score: {f1:.4f}")
                
                # Show convergence info for MLP
                if self.model_type == 'mlp' and hasattr(self.model, 'n_iter_'):
                    print(f"Converged after: {self.model.n_iter_} iterations")
                    print(f"Final loss: {self.model.loss_:.6f}")
                
                print("\nDetailed Classification Report:")
                print(classification_report(y_test, y_pred))
            
            # Log metrics
            logging.info(f"{self.model_type.upper()} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
            
            return accuracy, f1
            
        except Exception as e:
            logging.error(f"Error evaluating {self.model_type}: {str(e)}")
            raise
    
    def get_model_info(self):
        """
        Get detailed information about the trained model.
        
        Returns:
            dict: Model information and parameters
        """
        if not self.is_trained:
            return {"status": "not_trained"}
        
        info = {
            "model_type": self.model_type,
            "training_time": self.training_time,
            "parameters": self.model.get_params(),
            "is_scaled": self.scaler is not None
        }
        
        # Add MLP-specific info
        if self.model_type == 'mlp' and hasattr(self.model, 'n_iter_'):
            info.update({
                "iterations": self.model.n_iter_,
                "final_loss": getattr(self.model, 'loss_', None),
                "converged": self.model.n_iter_ < self.model.max_iter
            })
        
        return info
