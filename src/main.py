import pandas as pd
import logging
import os
import time
import numpy as np
from scipy import sparse
import joblib
from utils.nltk_ensure import ensure_nltk_resources
from preprocessing import Preprocessor
from representation import Representation
from features import get_features
from textclassification import TextClassification
import json
import sys

# Global variables
raw_data = None
classification_results = []
methods = ['bow', 'tfidf', 'word_emb', 'sentence_emb']
representation_metadata = {}

def setup_logging():
    """Initialize logging configuration"""
    try:
        log_dir = 'logs'
        os.makedirs(log_dir, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'{log_dir}/agnews_{time.strftime("%Y%m%d_%H%M%S")}.log'),
                logging.StreamHandler()
            ]
        )
        logging.info("Logging configuration initialized successfully")
        return True
    except Exception as e:
        print(f"Error setting up logging: {str(e)}")
        return False

def ensure_directories():
    """Create necessary directories"""
    try:
        directories = ['data/processed', 'results', 'logs']
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
        logging.info("All necessary directories created/verified")
        return True
    except Exception as e:
        logging.error(f"Error creating directories: {str(e)}")
        return False

def load_data():
    """Load AG News dataset"""
    global raw_data
    try:
        logging.info("Loading AG News dataset...")
        if not os.path.exists('data/raw/ag.news.csv'):
            logging.error("Dataset file 'data/raw/ag.news.csv' not found!")
            return False
            
        raw_data = pd.read_csv('data/raw/ag.news.csv', sep=',', encoding='utf-8')
        logging.info(f"Loaded dataset with {len(raw_data)} rows and {raw_data.shape[1]} columns")
        return True
    except Exception as e:
        logging.error(f"Error loading dataset: {str(e)}")
        return False

def run_preprocessing(selected_methods=None):
    """Run preprocessing for specified methods"""
    global raw_data
    
    if raw_data is None:
        logging.error("No data loaded. Please load data first.")
        return False
        
    if selected_methods is None:
        selected_methods = methods
        
    try:
        logging.info("Starting preprocessing phase...")
        ensure_nltk_resources()
        
        preprocessor = Preprocessor(raw_data)
        save_path = 'data/processed'
        
        successful_methods = []
        failed_methods = []
        
        for method in selected_methods:
            try:
                logging.info(f"Starting preprocessing with method: {method}")
                start_time = time.time()
                
                result = preprocessor.preprocess(method=method)
                
                processing_time = time.time() - start_time
                logging.info(f"Preprocessing with {method} completed in {processing_time:.2f} seconds")
                logging.info(f"Processed data shape: {result.shape}")
                
                # Save results
                output_file = f'{save_path}/news_cleaned_{method}.csv'
                result.to_csv(output_file, sep=",")
                logging.info(f"Saved processed data to {output_file}")
                successful_methods.append(method)
                
            except Exception as e:
                logging.error(f"Error processing method {method}: {str(e)}")
                failed_methods.append(method)
                continue
        
        logging.info(f"Preprocessing completed. Successful: {successful_methods}, Failed: {failed_methods}")
        return len(successful_methods) > 0
        
    except Exception as e:
        logging.error(f"Error in preprocessing phase: {str(e)}")
        return False

def run_representations(selected_methods=None):
    """Create representations for specified methods"""
    global representation_metadata
    
    if selected_methods is None:
        selected_methods = methods
        
    try:
        logging.info("Starting representation phase...")
        results_path = 'results'
        save_path = 'data/processed'
        
        successful_methods = []
        failed_methods = []
        
        for method in selected_methods:
            try:
                # Check if preprocessed file exists
                preprocessed_file = f'{save_path}/news_cleaned_{method}.csv'
                if not os.path.exists(preprocessed_file):
                    logging.error(f"Preprocessed file for {method} not found. Run preprocessing first.")
                    failed_methods.append(method)
                    continue
                
                logging.info(f"Creating {method} representation...")
                start_time = time.time()
                
                # Load preprocessed data
                preprocessed_df = pd.read_csv(preprocessed_file)
                logging.info(f"Loaded preprocessed data with shape: {preprocessed_df.shape}")
                
                # Create representation
                representation = Representation(preprocessed_df, method=method)
                X, model_or_vectorizer = representation.create_representation()
                
                # Log representation details
                processing_time = time.time() - start_time
                is_sparse = sparse.issparse(X)
                feature_type = "sparse" if is_sparse else "dense"
                
                logging.info(f"{method} representation: {X.shape[0]} documents, {X.shape[1]} features ({feature_type})")
                logging.info(f"Representation created in {processing_time:.2f} seconds")
                
                # Save representations
                if is_sparse:
                    sparse.save_npz(f'{results_path}/{method}_features.npz', X)
                else:
                    np.save(f'{results_path}/{method}_features.npy', X)
                
                # Save vectorizer or model
                if method in ['bow', 'tfidf']:
                    joblib.dump(model_or_vectorizer, f'{results_path}/{method}_vectorizer.pkl')
                elif method in ['word_emb', 'sentence_emb']:
                    model_or_vectorizer.save(f'{results_path}/{method}_model')
                
                logging.info(f"Saved {method} representation to {results_path}")
                
                # Store metadata for classification phase
                representation_metadata[method] = {
                    'feature_shape': list(X.shape),
                    'feature_type': feature_type,
                    'processing_time': processing_time
                }
                
                successful_methods.append(method)
                
            except Exception as e:
                logging.error(f"Error creating representation for {method}: {str(e)}")
                failed_methods.append(method)
                continue
        
        logging.info(f"Representation phase completed. Successful: {successful_methods}, Failed: {failed_methods}")
        return len(successful_methods) > 0
        
    except Exception as e:
        logging.error(f"Error in representation phase: {str(e)}")
        return False

def run_classifications(selected_methods=None):
    """Run classification for specified methods"""
    global classification_results, representation_metadata
    
    if selected_methods is None:
        selected_methods = methods
        
    try:
        logging.info("Starting classification phase...")
        
        successful_methods = []
        failed_methods = []
        
        for method in selected_methods:
            try:
                # Check if representation exists
                if method not in representation_metadata:
                    logging.error(f"No representation metadata found for {method}. Run representations first.")
                    failed_methods.append(method)
                    continue
                
                logging.info(f"Starting classification for {method}...")
                
                X_train, X_test, y_train, y_test = get_features(method)
                
                classification_start = time.time()
                
                # Train KNN
                logging.info(f"Training KNN for {method}...")
                knn_start = time.time()
                knn_classifier = TextClassification(model_type='knn')  # Use optimized defaults
                knn_classifier.fit(X_train, y_train)
                knn_acc, knn_f1 = knn_classifier.evaluate(X_test, y_test, verbose=False)  # Reduce output noise
                knn_time = time.time() - knn_start
                
                # Train MLP
                logging.info(f"Training MLP for {method}...")
                mlp_start = time.time()
                mlp_classifier = TextClassification(model_type='mlp')  # Use optimized defaults
                mlp_classifier.fit(X_train, y_train)
                mlp_acc, mlp_f1 = mlp_classifier.evaluate(X_test, y_test, verbose=False)  # Reduce output noise
                mlp_time = time.time() - mlp_start
                
                total_classification_time = time.time() - classification_start
                
                # Get representation metadata
                metadata = representation_metadata[method]
                
                # Store results
                method_results = {
                    'method': method,
                    'feature_shape': metadata['feature_shape'],
                    'feature_type': metadata['feature_type'],
                    'train_size': len(X_train),
                    'test_size': len(X_test),
                    'processing_time': metadata['processing_time'],
                    'classification_time': total_classification_time,
                    'knn': {
                        'accuracy': float(knn_acc),
                        'f1_score': float(knn_f1),
                        'training_time': float(knn_time),
                        'hyperparameters': knn_classifier.get_model_info().get('parameters', {})
                    },
                    'mlp': {
                        'accuracy': float(mlp_acc),
                        'f1_score': float(mlp_f1),
                        'training_time': float(mlp_time),
                        'hyperparameters': mlp_classifier.get_model_info().get('parameters', {})
                    },
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                }
                
                classification_results.append(method_results)
                
                logging.info(f"Results {method} - KNN: Acc={knn_acc:.4f}, F1={knn_f1:.4f}")
                logging.info(f"Results {method} - MLP: Acc={mlp_acc:.4f}, F1={mlp_f1:.4f}")
                
                successful_methods.append(method)
                
            except Exception as e:
                logging.error(f"Error in classification for {method}: {str(e)}")
                failed_methods.append(method)
                continue
        
        logging.info(f"Classification phase completed. Successful: {successful_methods}, Failed: {failed_methods}")
        return len(successful_methods) > 0
        
    except Exception as e:
        logging.error(f"Error in classification phase: {str(e)}")
        return False

def save_results():
    """Save classification results to JSON file"""
    global raw_data, classification_results
    
    try:
        if not classification_results:
            logging.warning("No classification results to save")
            return False
            
        results_path = 'results'
        
        results_summary = {
            'experiment_info': {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'dataset': 'AG News',
                'total_samples': len(raw_data) if raw_data is not None else 'unknown',
                'methods_tested': [result['method'] for result in classification_results],
                'classifiers': ['KNN', 'MLP']
            },
            'results': classification_results
        }
        
        # Save JSON file
        results_file = f'{results_path}/classification_results_{time.strftime("%Y%m%d_%H%M")}.json'
        os.makedirs(os.path.dirname(results_file), exist_ok=True)
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results_summary, f, indent=2, ensure_ascii=False)
        logging.info(f"Saved classification results to {results_file}")
        return True
        
    except Exception as e:
        logging.error(f"Error saving results: {str(e)}")
        return False

def show_results():
    """Display performance summary"""
    global classification_results
    
    try:
        if not classification_results:
            print("\nNo classification results available. Run classifications first.")
            return
        
        print("\n" + "="*60)
        print("PERFORMANCE SUMMARY")
        print("="*60)
        
        # Show results for each method
        for result in classification_results:
            method = result['method']
            print(f"\n{method.upper()} Results:")
            print(f"  Features: {result['feature_shape']} ({result['feature_type']})")
            print(f"  KNN - Accuracy: {result['knn']['accuracy']:.4f}, F1: {result['knn']['f1_score']:.4f}")
            print(f"  MLP - Accuracy: {result['mlp']['accuracy']:.4f}, F1: {result['mlp']['f1_score']:.4f}")
        
        # Best performance
        if len(classification_results) > 1:
            best_knn = max(classification_results, key=lambda x: x['knn']['accuracy'])
            best_mlp = max(classification_results, key=lambda x: x['mlp']['accuracy'])
            
            print(f"\nBEST PERFORMANCE:")
            print(f"  Best KNN: {best_knn['method']} with {best_knn['knn']['accuracy']:.4f} accuracy")
            print(f"  Best MLP: {best_mlp['method']} with {best_mlp['mlp']['accuracy']:.4f} accuracy")
        
        print("="*60)
        
    except Exception as e:
        logging.error(f"Error displaying results: {str(e)}")
        print(f"Error displaying results: {str(e)}")

def run_full_pipeline():
    """Run the complete pipeline"""
    try:
        print("Starting full pipeline...")
        
        # Load data
        if not load_data():
            return False
        
        # Run preprocessing
        if not run_preprocessing():
            return False
        
        # Run representations
        if not run_representations():
            return False
        
        # Run classifications
        if not run_classifications():
            return False
        
        # Save results
        if not save_results():
            return False
        
        print("Full pipeline completed successfully!")
        return True
        
    except Exception as e:
        logging.error(f"Error in full pipeline: {str(e)}")
        print(f"Error in full pipeline: {str(e)}")
        return False

def select_methods():
    """Allow user to select specific methods"""
    print("\nAvailable methods:")
    for i, method in enumerate(methods, 1):
        print(f"{i}. {method}")
    print(f"{len(methods)+1}. All methods")
    
    try:
        choice = input("\nSelect methods (comma-separated numbers): ").strip()
        if not choice:
            return methods
        
        selected_indices = [int(x.strip()) for x in choice.split(',')]
        
        if len(methods)+1 in selected_indices:
            return methods
        
        selected_methods = []
        for idx in selected_indices:
            if 1 <= idx <= len(methods):
                selected_methods.append(methods[idx-1])
        
        return selected_methods if selected_methods else methods
        
    except Exception as e:
        print(f"Invalid selection, using all methods: {str(e)}")
        return methods

def display_menu():
    """Display CLI menu options"""
    print("\n" + "="*60)
    print("AG NEWS TEXT CLASSIFICATION PIPELINE")
    print("="*60)
    print("1. Load Dataset")
    print("2. Run Preprocessing")
    print("3. Create Representations")
    print("4. Run Classifications")
    print("5. Save Results")
    print("6. Show Results Summary")
    print("7. Run Full Pipeline")
    print("8. Exit")
    print("="*60)

def main():
    """Main function with CLI interface"""
    print("Welcome to the AG News Text Classification System!")
    
    # Setup
    if not setup_logging():
        print("Failed to setup logging. Exiting.")
        return
    
    if not ensure_directories():
        print("Failed to create necessary directories. Exiting.")
        return
    
    logging.info("System initialized successfully")
    
    # Main program loop
    while True:
        try:
            display_menu()
            choice = input("\nEnter your choice (1-8): ").strip()
            
            if choice == '1':
                print("\nLoading dataset...")
                if load_data():
                    print("Dataset loaded successfully!")
                else:
                    print("Failed to load dataset.")
            
            elif choice == '2':
                selected_methods = select_methods()
                print(f"\nRunning preprocessing for: {selected_methods}")
                if run_preprocessing(selected_methods):
                    print("Preprocessing completed!")
                else:
                    print("Preprocessing failed.")
            
            elif choice == '3':
                selected_methods = select_methods()
                print(f"\nCreating representations for: {selected_methods}")
                if run_representations(selected_methods):
                    print("Representations created!")
                else:
                    print("Representation creation failed.")
            
            elif choice == '4':
                selected_methods = select_methods()
                print(f"\nRunning classifications for: {selected_methods}")
                if run_classifications(selected_methods):
                    print("Classifications completed!")
                else:
                    print("Classifications failed.")
            
            elif choice == '5':
                print("\nSaving results...")
                if save_results():
                    print("Results saved successfully!")
                else:
                    print("Failed to save results.")
            
            elif choice == '6':
                show_results()
            
            elif choice == '7':
                if run_full_pipeline():
                    show_results()
                else:
                    print("Full pipeline failed.")
            
            elif choice == '8':
                print("\nGoodbye!")
                logging.info("System shutdown")
                break
            
            else:
                print("Invalid choice. Please enter 1-8.")
            
            input("\nPress Enter to continue...")
            
        except KeyboardInterrupt:
            print("\n\nInterrupted by user. Goodbye!")
            logging.info("System interrupted by user")
            break
        except Exception as e:
            logging.error(f"Unexpected error in main loop: {str(e)}")
            print(f"An unexpected error occurred: {str(e)}")
            input("Press Enter to continue...")

if __name__ == "__main__":
    main()

