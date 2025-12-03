#!/usr/bin/env python3
"""
============================================================================
Ensemble Anomaly Detection Model
============================================================================
Author: Parth Saraykar
Purpose: Combine Isolation Forest, LOF, and One-Class SVM
Strategy: Weighted voting based on individual model ROC-AUC performance
Expected: ROC-AUC ≥ 0.85, Precision@10% ≥ 0.80
============================================================================
"""

import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
import logging
import sys
from typing import Dict, List, Tuple, Optional
from datetime import datetime

from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.preprocessing import StandardScaler

import mlflow
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================================
# LOGGER SETUP
# ============================================================================

def setup_logger():
    """Setup logger"""
    Path("logs").mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'logs/ensemble_model_{timestamp}.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


# ============================================================================
# ENSEMBLE MODEL CLASS
# ============================================================================

class EnsembleAnomalyDetector:
    """Ensemble of multiple anomaly detection models"""
    
    def __init__(self, logger: logging.Logger = None):
        self.logger = logger or logging.getLogger(__name__)
        self.models = {}
        self.weights = {}
        self.scaler = None
        self.feature_names = []
    
    def load_trained_models(self, model_dir: str = "models/anomaly_detection"):
        """Load all trained models and their artifacts"""
        self.logger.info("Loading trained models...")
        
        model_names = ['Isolation_Forest', 'LOF', 'One_Class_SVM']
        loaded_count = 0
        
        for name in model_names:
            model_path = Path(model_dir) / name / "model.pkl"
            
            if model_path.exists():
                try:
                    with open(model_path, 'rb') as f:
                        self.models[name] = pickle.load(f)
                    self.logger.info(f"  ✓ Loaded {name}")
                    loaded_count += 1
                except Exception as e:
                    self.logger.warning(f"  ✗ Failed to load {name}: {str(e)}")
            else:
                self.logger.warning(f"  ✗ Model not found: {model_path}")
        
        if loaded_count == 0:
            raise FileNotFoundError("No models could be loaded. Train models first.")
        
        # Load scaler (use from One_Class_SVM as it's usually the best)
        scaler_path = Path(model_dir) / "One_Class_SVM" / "scaler.pkl"
        if scaler_path.exists():
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            self.logger.info(f"  ✓ Loaded scaler")
        
        # Load feature names
        features_path = Path(model_dir) / "One_Class_SVM" / "features.json"
        if features_path.exists():
            with open(features_path, 'r') as f:
                self.feature_names = json.load(f)['features']
            self.logger.info(f"  ✓ Loaded {len(self.feature_names)} feature names")
        
        self.logger.info(f"\n✓ Successfully loaded {loaded_count} models")
    
    def set_weights_by_performance(self, performance_scores: Dict[str, float]):
        """
        Set ensemble weights based on validation performance
        
        Args:
            performance_scores: Dict of {model_name: roc_auc_score}
        """
        self.logger.info("\nSetting ensemble weights...")
        
        # Only use models that were actually loaded
        available_scores = {name: score for name, score in performance_scores.items() 
                           if name in self.models}
        
        # Normalize scores to sum to 1 (softmax-like)
        total = sum(available_scores.values())
        self.weights = {name: score/total for name, score in available_scores.items()}
        
        self.logger.info("Weights (based on ROC-AUC):")
        for name, weight in sorted(self.weights.items(), key=lambda x: x[1], reverse=True):
            self.logger.info(f"  {name:20s}: {weight:.4f} (ROC-AUC: {available_scores[name]:.4f})")
    
    def set_equal_weights(self):
        """Set equal weights for all models"""
        n_models = len(self.models)
        self.weights = {name: 1.0/n_models for name in self.models.keys()}
        self.logger.info(f"\nUsing equal weights: {1.0/n_models:.4f} each")
    
    def predict(self, X):
        """
        Ensemble prediction using weighted voting
        
        Args:
            X: Input features (scaled)
            
        Returns:
            Tuple of (binary_predictions, anomaly_scores)
        """
        predictions = {}
        scores = {}
        
        # Get predictions from each model
        for name, model in self.models.items():
            try:
                # Predict
                pred = model.predict(X)
                
                # Get anomaly scores if available
                if hasattr(model, 'score_samples'):
                    score = model.score_samples(X)
                else:
                    score = pred  # Fallback for models without score_samples
                
                # Convert to binary (1 = anomaly, 0 = normal)
                predictions[name] = (pred == -1).astype(int)
                
                # Invert scores so higher = more anomalous
                scores[name] = -score
                
            except Exception as e:
                self.logger.warning(f"Error predicting with {name}: {str(e)}")
                continue
        
        # Weighted voting on predictions
        ensemble_pred = np.zeros(len(X))
        for name, pred in predictions.items():
            weight = self.weights.get(name, 1.0/len(predictions))
            ensemble_pred += pred * weight
        
        # Threshold at 0.5 for binary prediction
        ensemble_pred_binary = (ensemble_pred >= 0.5).astype(int)
        
        # Weighted average of anomaly scores
        ensemble_scores = np.zeros(len(X))
        for name, score in scores.items():
            weight = self.weights.get(name, 1.0/len(scores))
            ensemble_scores += score * weight
        
        return ensemble_pred_binary, ensemble_scores
    
    def save(self, output_path: str = "models/anomaly_detection/Ensemble"):
        """Save ensemble model artifacts"""
        output_dir = Path(output_path)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Save ensemble object
        with open(output_dir / 'ensemble.pkl', 'wb') as f:
            pickle.dump(self, f)
        
        # Save weights separately
        with open(output_dir / 'weights.json', 'w') as f:
            json.dump(self.weights, f, indent=2)
        
        # Save feature names
        with open(output_dir / 'features.json', 'w') as f:
            json.dump({'features': self.feature_names}, f, indent=2)
        
        self.logger.info(f"✓ Ensemble artifacts saved to: {output_dir}")


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_ensemble(y_true, y_pred, scores, logger):
    """Comprehensive ensemble evaluation"""
    
    metrics = {}
    
    # ROC-AUC
    try:
        metrics['roc_auc'] = roc_auc_score(y_true, scores)
    except:
        metrics['roc_auc'] = 0.0
    
    # Standard metrics
    metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
    metrics['f1_score'] = f1_score(y_true, y_pred, zero_division=0)
    
    # Precision@10%
    k = int(len(y_true) * 0.10)
    top_k_idx = np.argsort(-scores)[:k]
    top_k_preds = np.zeros(len(y_true))
    top_k_preds[top_k_idx] = 1
    
    metrics['precision_at_10pct'] = precision_score(y_true, top_k_preds, zero_division=0)
    metrics['recall_at_10pct'] = recall_score(y_true, top_k_preds, zero_division=0)
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics['true_positives'] = int(tp)
    metrics['false_positives'] = int(fp)
    metrics['true_negatives'] = int(tn)
    metrics['false_negatives'] = int(fn)
    
    return metrics


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_ensemble_comparison(individual_results, ensemble_metrics, logger):
    """Compare ensemble vs individual models"""
    logger.info("\nCreating ensemble comparison plot...")
    
    Path("outputs/models/plots").mkdir(exist_ok=True, parents=True)
    
    # Prepare data
    model_names = list(individual_results.keys()) + ['Ensemble']
    roc_aucs = [individual_results[m]['roc_auc'] for m in individual_results.keys()] + [ensemble_metrics['roc_auc']]
    precisions_at_k = [individual_results[m]['precision_at_10pct'] for m in individual_results.keys()] + [ensemble_metrics['precision_at_10pct']]
    
    # Create plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # ROC-AUC comparison
    colors = ['steelblue', 'coral', 'lightgreen', 'gold']
    axes[0].bar(model_names, roc_aucs, color=colors, edgecolor='black', linewidth=1.5)
    axes[0].axhline(y=0.85, color='red', linestyle='--', linewidth=2, label='Target: 0.85')
    axes[0].set_ylabel('ROC-AUC', fontsize=12, fontweight='bold')
    axes[0].set_title('ROC-AUC: Individual Models vs Ensemble', fontsize=14, fontweight='bold')
    axes[0].set_ylim([0, 1])
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3, axis='y')
    axes[0].tick_params(axis='x', rotation=45)
    
    # Annotate values
    for i, (name, value) in enumerate(zip(model_names, roc_aucs)):
        axes[0].text(i, value + 0.02, f'{value:.3f}', ha='center', fontsize=10, fontweight='bold')
    
    # Precision@10% comparison
    axes[1].bar(model_names, precisions_at_k, color=colors, edgecolor='black', linewidth=1.5)
    axes[1].axhline(y=0.80, color='red', linestyle='--', linewidth=2, label='Target: 0.80')
    axes[1].set_ylabel('Precision@10%', fontsize=12, fontweight='bold')
    axes[1].set_title('Precision@10%: Individual Models vs Ensemble', fontsize=14, fontweight='bold')
    axes[1].set_ylim([0, 1])
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].tick_params(axis='x', rotation=45)
    
    # Annotate values
    for i, (name, value) in enumerate(zip(model_names, precisions_at_k)):
        axes[1].text(i, value + 0.02, f'{value:.3f}', ha='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    output_path = Path("outputs/models/plots/ensemble_vs_individual.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✓ Saved: {output_path}")
    
    return output_path


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Train and evaluate ensemble model"""
    
    logger = setup_logger()
    
    logger.info("="*80)
    logger.info("ENSEMBLE ANOMALY DETECTION MODEL")
    logger.info("="*80)
    logger.info("Strategy: Weighted voting of IF + LOF + One-Class SVM")
    logger.info("Weights: Based on individual model ROC-AUC scores")
    logger.info("="*80)
    logger.info("")
    
    # Load data
    logger.info("Loading validation data...")
    
    # Try enhanced data first
    try:
        df = pd.read_csv('outputs/snorkel/data/snorkel_labeled_enhanced.csv')
        logger.info("✓ Using ENHANCED dataset")
    except:
        df = pd.read_csv('outputs/snorkel/data/snorkel_labeled_only.csv')
        logger.info("✓ Using standard dataset")
    
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    
    # Validation set
    val_df = df[(df['Year'] >= 2019) & (df['Year'] <= 2021)].copy()
    logger.info(f"Validation samples: {len(val_df):,}")
    
    # Load feature names
    try:
        with open('models/anomaly_detection/One_Class_SVM/features.json', 'r') as f:
            feature_cols = json.load(f)['features']
        logger.info(f"✓ Loaded {len(feature_cols)} features")
    except:
        logger.error("❌ Could not load features. Train models first.")
        sys.exit(1)
    
    # Add new interaction features if they exist
    new_features = [col for col in df.columns if any(x in col for x in 
                   ['_x_', 'vulnerability', 'composite', 'volatility', 'acceleration', 
                    'signal_count', 'stress_flag', 'ratio', 'momentum'])]
    
    for feat in new_features:
        if feat not in feature_cols and feat in df.columns:
            feature_cols.append(feat)
    
    logger.info(f"Using {len(feature_cols)} total features (including {len(new_features)} new)")
    
    # Prepare features
    X_val = val_df[feature_cols].fillna(val_df[feature_cols].median())
    X_val = X_val.replace([np.inf, -np.inf], np.nan).fillna(X_val.median())
    y_val = val_df['snorkel_label'].values
    
    logger.info(f"Val at-risk rate: {y_val.sum()/len(y_val):.2%}")
    
    # Load scaler
    try:
        with open('models/anomaly_detection/One_Class_SVM/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        X_val_scaled = scaler.transform(X_val)
        logger.info("✓ Features scaled\n")
    except:
        logger.warning("⚠ Could not load scaler, using unscaled features")
        X_val_scaled = X_val.values
    
    # ========================================================================
    # CREATE ENSEMBLE
    # ========================================================================
    
    logger.info("="*80)
    logger.info("CREATING ENSEMBLE MODEL")
    logger.info("="*80)
    
    ensemble = EnsembleAnomalyDetector(logger)
    ensemble.load_trained_models()
    
    # Set weights based on individual model performance
    # These are the ROC-AUC scores from initial training
    individual_performance = {
        'Isolation_Forest': 0.7818,
        'LOF': 0.7737,
        'One_Class_SVM': 0.8173
    }
    
    # Only use weights for models that were loaded
    loaded_performance = {name: score for name, score in individual_performance.items() 
                         if name in ensemble.models}
    
    ensemble.set_weights_by_performance(loaded_performance)
    
    # ========================================================================
    # EVALUATE ENSEMBLE
    # ========================================================================
    
    logger.info("\n" + "="*80)
    logger.info("ENSEMBLE PREDICTION & EVALUATION")
    logger.info("="*80)
    
    # Predict
    y_pred_ensemble, scores_ensemble = ensemble.predict(X_val_scaled)
    
    # Evaluate
    metrics = evaluate_ensemble(y_val, y_pred_ensemble, scores_ensemble, logger)
    
    logger.info("\nEnsemble Performance:")
    logger.info(f"  ROC-AUC:         {metrics['roc_auc']:.4f} (target: ≥0.85) {'✓ PASS' if metrics['roc_auc'] >= 0.85 else '✗ FAIL'}")
    logger.info(f"  Precision@10%:   {metrics['precision_at_10pct']:.4f} (target: ≥0.80) {'✓ PASS' if metrics['precision_at_10pct'] >= 0.80 else '✗ FAIL'}")
    logger.info(f"  Precision:       {metrics['precision']:.4f}")
    logger.info(f"  Recall:          {metrics['recall']:.4f}")
    logger.info(f"  F1-Score:        {metrics['f1_score']:.4f}")
    
    # ========================================================================
    # LOG TO MLFLOW
    # ========================================================================
    
    logger.info("\n" + "="*80)
    logger.info("LOGGING TO MLFLOW")
    logger.info("="*80)
    
    mlflow.set_experiment("financial_stress_model3_anomaly_detection")
    
    with mlflow.start_run(run_name="ensemble_weighted_voting") as run:
        
        # Log ensemble configuration
        mlflow.log_params({
            'ensemble_type': 'weighted_voting',
            'n_models': len(ensemble.models),
            'models': ','.join(ensemble.models.keys())
        })
        
        # Log weights
        for name, weight in ensemble.weights.items():
            mlflow.log_param(f'weight_{name}', weight)
        
        # Log metrics
        mlflow.log_metrics(metrics)
        
        # Log tags
        mlflow.set_tags({
            'model': 'Ensemble',
            'component_models': ','.join(ensemble.models.keys()),
            'author': 'Parth_Saraykar',
            'project': 'Financial_Stress_Test'
        })
        
        # Save ensemble
        ensemble.save()
        mlflow.log_artifact('models/anomaly_detection/Ensemble/ensemble.pkl')
        mlflow.log_artifact('models/anomaly_detection/Ensemble/weights.json')
        
        run_id = run.info.run_id
        logger.info(f"✓ Logged to MLflow (Run ID: {run_id})")
    
    # ========================================================================
    # VISUALIZATIONS
    # ========================================================================
    
    logger.info("\n" + "="*80)
    logger.info("CREATING VISUALIZATIONS")
    logger.info("="*80)
    
    # Prepare individual model results for comparison
    individual_results = {}
    for name, perf in loaded_performance.items():
        individual_results[name] = {
            'roc_auc': perf,
            'precision_at_10pct': 0.67 if name == 'One_Class_SVM' else 0.44  # Approximate from training
        }
    
    plot_path = plot_ensemble_comparison(individual_results, metrics, logger)
    mlflow.log_artifact(str(plot_path))
    
    # ========================================================================
    # FINAL REPORT
    # ========================================================================
    
    logger.info("\n" + "="*80)
    logger.info("GENERATING ENSEMBLE REPORT")
    logger.info("="*80)
    
    report = f"""
{'='*80}
ENSEMBLE MODEL REPORT
{'='*80}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Author: Parth Saraykar

ENSEMBLE CONFIGURATION
{'='*80}
Strategy: Weighted Voting
Component Models: {len(ensemble.models)}
"""
    
    for name, weight in ensemble.weights.items():
        report += f"  - {name:20s}: weight={weight:.4f}\n"
    
    report += f"""
ENSEMBLE PERFORMANCE
{'='*80}
ROC-AUC:           {metrics['roc_auc']:.4f} {'✓ PASS' if metrics['roc_auc'] >= 0.85 else '✗ FAIL'} (target: ≥0.85)
Precision@10%:     {metrics['precision_at_10pct']:.4f} {'✓ PASS' if metrics['precision_at_10pct'] >= 0.80 else '✗ FAIL'} (target: ≥0.80)
Recall@10%:        {metrics['recall_at_10pct']:.4f}
F1-Score:          {metrics['f1_score']:.4f}
Precision:         {metrics['precision']:.4f}
Recall:            {metrics['recall']:.4f}

CONFUSION MATRIX
{'='*80}
True Positives:    {metrics['true_positives']}
False Positives:   {metrics['false_positives']}
True Negatives:    {metrics['true_negatives']}
False Negatives:   {metrics['false_negatives']}

COMPARISON TO BEST INDIVIDUAL MODEL
{'='*80}
Best Individual (One-Class SVM):
  ROC-AUC: 0.8173
  Precision@10%: 0.6700

Ensemble:
  ROC-AUC: {metrics['roc_auc']:.4f} ({metrics['roc_auc']-0.8173:+.4f})
  Precision@10%: {metrics['precision_at_10pct']:.4f} ({metrics['precision_at_10pct']-0.67:+.4f})

OUTPUT ARTIFACTS
{'='*80}
✓ Ensemble model: models/anomaly_detection/Ensemble/ensemble.pkl
✓ Weights: models/anomaly_detection/Ensemble/weights.json
✓ Comparison plot: outputs/models/plots/ensemble_vs_individual.png
✓ MLflow Run ID: {run_id}

{'='*80}
END OF REPORT
{'='*80}
"""
    
    report_path = Path("outputs/models/reports/ensemble_report.txt")
    with open(report_path, 'w') as f:
        f.write(report)
    
    logger.info(f"✓ Report saved: {report_path}")
    
    print("\n" + report)
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    
    logger.info("\n" + "="*80)
    logger.info("✓ ENSEMBLE MODEL CREATION COMPLETE!")
    logger.info("="*80)
    logger.info("\nKey Results:")
    logger.info(f"  ROC-AUC: {metrics['roc_auc']:.4f} {'✓' if metrics['roc_auc'] >= 0.85 else '(close!)'}")
    logger.info(f"  Precision@10%: {metrics['precision_at_10pct']:.4f} {'✓' if metrics['precision_at_10pct'] >= 0.80 else '(close!)'}")
    logger.info("\nNext steps:")
    logger.info("  1. View MLflow UI: mlflow ui --port 5000")
    logger.info("  2. Check ensemble report: cat outputs/models/reports/ensemble_report.txt")
    logger.info("  3. Integrate with teammates' models")


if __name__ == "__main__":
    main()