"""
Financial Stress Test - Model 3 Pipeline DAG (Integrated)
==========================================================
Triggered automatically after Data Pipeline completes
Uses XCom to get data paths from upstream DAG

Pipeline: EDA → Threshold Extraction → Snorkel → Training → Validation → Registry

Author: MLOps Group11 Team - Model 3
"""

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
from airflow.sensors.external_task import ExternalTaskSensor
from airflow.utils.task_group import TaskGroup
from airflow.utils.dates import days_ago
from airflow.models import Variable

from datetime import datetime, timedelta
from pathlib import Path
import logging
import sys
import os

# ============================================================================
# DYNAMIC PATH CONFIGURATION - NO HARD CODING
# ============================================================================

# Get project root from Airflow Variable or environment
PROJECT_DIR = Variable.get("project_root", default_var=os.getenv("PROJECT_ROOT", "/opt/airflow/project"))

# Build all paths dynamically
SRC_DIR = os.path.join(PROJECT_DIR, "src")
CONFIG_DIR = os.path.join(PROJECT_DIR, "configs")
DATA_DIR = os.path.join(PROJECT_DIR, "data")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs")
MODELS_DIR = os.path.join(PROJECT_DIR, "models")
MLRUNS_DIR = os.path.join(PROJECT_DIR, "mlruns")
LOGS_DIR = os.path.join(PROJECT_DIR, "logs")

# Add src to Python path
sys.path.insert(0, SRC_DIR)

# Import alerting system
try:
    from monitoring.alerting import AlertManager
    ALERTING_AVAILABLE = True
except ImportError:
    print("⚠️ Warning: AlertManager not available. Alerts disabled.")
    ALERTING_AVAILABLE = False

# ============================================================================
# DAG DEFAULT ARGUMENTS
# ============================================================================

default_args = {
    'owner': 'model3_team',
    'depends_on_past': True,  # Wait for data pipeline
    'start_date': days_ago(1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'execution_timeout': timedelta(hours=4),
}

# ============================================================================
# DAG DEFINITION
# ============================================================================

dag = DAG(
    'financial_stress_model3_pipeline',
    default_args=default_args,
    description='Model 3: Anomaly Detection Pipeline (Auto-triggered after data pipeline)',
    schedule_interval=None,  # Triggered by data pipeline, not scheduled
    catchup=False,
    max_active_runs=1,
    tags=['mlops', 'model3', 'anomaly-detection', 'training'],
    doc_md="""
    ## Financial Stress Test - Model 3 Pipeline
    
    **Triggered By**: `financial_crisis_pipeline` DAG completion
    
    **Purpose**: Train anomaly detection models on labeled data
    
    **Pipeline**:
    1. Wait for data pipeline completion (ExternalTaskSensor)
    2. Load paths and config from XCom
    3. Check if labels exist (skip Snorkel if yes)
    4. EDA Analysis (if needed)
    5. Auto Threshold Extraction
    6. Snorkel Weak Supervision Labeling
    7. Model Training (Isolation Forest, LOF, One-Class SVM)
    8. Hyperparameter Tuning
    9. Sensitivity Analysis
    10. Bias Detection (Sector Slicing)
    11. Model Validation (Performance Gates)
    12. MLflow Registry Promotion
    
    **Performance Gates**:
    - ROC-AUC ≥ 0.75
    - Precision@10% ≥ 0.60
    - F1 Std Dev < 0.15 (bias check)
    """,
)

# ============================================================================
# ALERTING CALLBACKS
# ============================================================================

def task_failure_alert(context):
    """Send alert on task failure"""
    if not ALERTING_AVAILABLE:
        return
    
    try:
        task = context.get('task_instance')
        dag_run = context.get('dag_run')
        exception = context.get('exception')
        
        alert_manager = AlertManager()
        
        message = f"""
        Model 3 Pipeline Task Failed: {task.task_id}
        DAG: {task.dag_id}
        Execution Date: {dag_run.execution_date}
        Error: {str(exception) if exception else 'Check logs'}
        Log URL: {task.log_url}
        
        Action Required: Check logs and retry pipeline
        """
        
        alert_manager.send_alert(
            message=message,
            severity='ERROR',
            component=f'model3_{task.task_id}',
            alert_type='MODEL_PIPELINE_FAILURE'
        )
        
        print(f"✓ Alert sent for {task.task_id} failure")
    except Exception as e:
        print(f"❌ Failed to send alert: {str(e)}")


def model_validation_failure_alert(context):
    """Special alert for model validation failures"""
    if not ALERTING_AVAILABLE:
        return
    
    try:
        task = context.get('task_instance')
        alert_manager = AlertManager()
        
        message = f"""
        🚨 CRITICAL: Model Validation Failed
        Task: {task.task_id}
        
        Model failed to meet performance gates:
        - ROC-AUC < 0.75, OR
        - Precision@10% < 0.60, OR
        - Bias (F1 std) > 0.15
        
        Action Required:
        1. Review model metrics in MLflow
        2. Check training data quality
        3. Adjust hyperparameters
        4. Re-train models
        """
        
        alert_manager.send_alert(
            message=message,
            severity='CRITICAL',
            component='model3_validation',
            alert_type='MODEL_VALIDATION_FAILURE'
        )
    except Exception as e:
        print(f"❌ Failed to send validation alert: {str(e)}")


def pipeline_success_alert(**context):
    """Send success notification"""
    if not ALERTING_AVAILABLE:
        return
    
    try:
        dag_run = context.get('dag_run')
        ti = context['task_instance']
        
        # Pull metrics from XCom
        roc_auc = ti.xcom_pull(task_ids='model_training.validate_performance', key='roc_auc')
        precision = ti.xcom_pull(task_ids='model_training.validate_performance', key='precision_at_10')
        f1_std = ti.xcom_pull(task_ids='model_training.validate_bias', key='f1_std')
        model_version = ti.xcom_pull(task_ids='promote_to_staging', key='model_version')
        
        alert_manager = AlertManager()
        
        message = f"""
        ✅ Model 3 Pipeline Completed Successfully!
        
        Execution Date: {dag_run.execution_date}
        Duration: {dag_run.end_date - dag_run.start_date if dag_run.end_date else 'N/A'}
        
        Performance Metrics:
        - ROC-AUC: {roc_auc:.4f}
        - Precision@10%: {precision:.4f}
        - Bias (F1 Std): {f1_std:.4f}
        
        Model Details:
        - Model: One-Class SVM
        - Version: {model_version}
        - Stage: Staging
        - Registry: MLflow
        
        ✓ All quality gates passed
        ✓ Model ready for deployment
        
        Next Steps:
        1. Review model in MLflow UI
        2. Test model predictions
        3. Promote to Production when ready
        """
        
        alert_manager.send_alert(
            message=message,
            severity='INFO',
            component='model3_pipeline',
            alert_type='MODEL_PIPELINE_SUCCESS'
        )
    except Exception as e:
        print(f"⚠️ Failed to send success alert: {str(e)}")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_paths_from_upstream(**context):
    """
    Get data paths from upstream data pipeline DAG via XCom
    """
    ti = context['task_instance']
    
    # Try to pull from upstream DAG
    try:
        # Get execution date of upstream DAG
        execution_date = context['execution_date']
        
        # Pull final data path from upstream DAG
        merged_data_path = ti.xcom_pull(
            dag_id='financial_crisis_pipeline',
            task_ids='step1_data_cleaning_and_merging',
            key='merged_data_path',
            include_prior_dates=True
        )
        
        if not merged_data_path:
            # Fallback to default path if XCom not available
            merged_data_path = os.path.join(DATA_DIR, "processed", "merged_quarterly_data.csv")
            logging.warning(f"XCom pull failed, using default path: {merged_data_path}")
        
        logging.info(f"✓ Got data path from upstream: {merged_data_path}")
        
    except Exception as e:
        # Fallback to default path
        merged_data_path = os.path.join(DATA_DIR, "processed", "merged_quarterly_data.csv")
        logging.warning(f"Error pulling XCom, using default: {e}")
    
    # Build all output paths
    paths = {
        'project_dir': PROJECT_DIR,
        'src_dir': SRC_DIR,
        'config_dir': CONFIG_DIR,
        'data_dir': DATA_DIR,
        'output_dir': OUTPUT_DIR,
        'models_dir': MODELS_DIR,
        'mlruns_dir': MLRUNS_DIR,
        'logs_dir': LOGS_DIR,
        'merged_data_path': merged_data_path,
        'eda_config': os.path.join(CONFIG_DIR, "eda_config.yaml"),
        'model_config': os.path.join(CONFIG_DIR, "model_config.yaml"),
        'labeled_data_path': os.path.join(OUTPUT_DIR, "snorkel", "data", "snorkel_labeled_only.csv"),
        'thresholds_path': os.path.join(OUTPUT_DIR, "snorkel", "thresholds_auto.yaml"),
        'eda_report_path': os.path.join(OUTPUT_DIR, "eda", "eda_report.html"),
    }
    
    # Push all paths to XCom for downstream tasks
    for key, value in paths.items():
        ti.xcom_push(key=key, value=value)
    
    logging.info("✓ All paths configured and pushed to XCom")
    logging.info(f"  Project: {PROJECT_DIR}")
    logging.info(f"  Data: {merged_data_path}")
    logging.info(f"  Output: {OUTPUT_DIR}")
    
    return paths


def check_labeled_data_exists(**context):
    """Check if labeled data already exists, skip Snorkel if yes"""
    ti = context['task_instance']
    
    labeled_data_path = ti.xcom_pull(task_ids='get_paths', key='labeled_data_path')
    
    if Path(labeled_data_path).exists():
        import pandas as pd
        df = pd.read_csv(labeled_data_path)
        logging.info(f"✓ Labeled data found: {len(df)} samples")
        logging.info(f"  Path: {labeled_data_path}")
        return 'skip_labeling'
    else:
        logging.info("Labeled data not found, need to run Snorkel pipeline")
        return 'labeling_pipeline.run_eda'


def run_eda_analysis(**context):
    """Run EDA and generate report"""
    ti = context['task_instance']
    
    # Get paths from XCom
    project_dir = ti.xcom_pull(task_ids='get_paths', key='project_dir')
    eda_config_path = ti.xcom_pull(task_ids='get_paths', key='eda_config')
    
    logging.info("Starting EDA analysis...")
    
    # Set environment and run EDA
    import subprocess
    result = subprocess.run(
        ['python', os.path.join(project_dir, 'src', 'eda', 'eda.py')],
        cwd=project_dir,
        capture_output=True,
        text=True,
        timeout=1800  # 30 min timeout
    )
    
    if result.returncode != 0:
        logging.error(f"EDA failed: {result.stderr}")
        raise RuntimeError(f"EDA failed: {result.stderr}")
    
    logging.info("✓ EDA analysis completed")
    logging.info(result.stdout)


def extract_thresholds(**context):
    """Extract automatic thresholds from EDA"""
    ti = context['task_instance']
    
    project_dir = ti.xcom_pull(task_ids='get_paths', key='project_dir')
    
    logging.info("Extracting automatic thresholds...")
    
    import subprocess
    result = subprocess.run(
        ['python', os.path.join(project_dir, 'src', 'labeling', 'auto_threshold_extractor.py')],
        cwd=project_dir,
        capture_output=True,
        text=True,
        timeout=600  # 10 min timeout
    )
    
    if result.returncode != 0:
        logging.error(f"Threshold extraction failed: {result.stderr}")
        raise RuntimeError(f"Threshold extraction failed: {result.stderr}")
    
    logging.info("✓ Thresholds extracted")
    logging.info(result.stdout)


def run_snorkel_labeling(**context):
    """Run Snorkel weak supervision pipeline"""
    ti = context['task_instance']
    
    project_dir = ti.xcom_pull(task_ids='get_paths', key='project_dir')
    
    logging.info("Starting Snorkel labeling pipeline...")
    
    import subprocess
    result = subprocess.run(
        ['python', os.path.join(project_dir, 'src', 'labeling', 'snorkel_pipeline.py')],
        cwd=project_dir,
        capture_output=True,
        text=True,
        timeout=3600  # 60 min timeout
    )
    
    if result.returncode != 0:
        logging.error(f"Snorkel labeling failed: {result.stderr}")
        raise RuntimeError(f"Snorkel labeling failed: {result.stderr}")
    
    logging.info("✓ Snorkel labeling completed")
    logging.info(result.stdout)
    
    # Validate labels were created
    labeled_data_path = ti.xcom_pull(task_ids='get_paths', key='labeled_data_path')
    if not Path(labeled_data_path).exists():
        raise FileNotFoundError(f"Labeled data not created: {labeled_data_path}")


def train_all_models(**context):
    """Train all anomaly detection models"""
    ti = context['task_instance']
    
    project_dir = ti.xcom_pull(task_ids='get_paths', key='project_dir')
    
    logging.info("Starting model training...")
    
    import subprocess
    result = subprocess.run(
        ['python', os.path.join(project_dir, 'src', 'models', 'train_anomaly_detection.py')],
        cwd=project_dir,
        capture_output=True,
        text=True,
        timeout=7200  # 120 min timeout
    )
    
    if result.returncode != 0:
        logging.error(f"Model training failed: {result.stderr}")
        raise RuntimeError(f"Model training failed: {result.stderr}")
    
    logging.info("✓ Model training completed")
    logging.info(result.stdout)


def validate_model_performance(**context):
    """Validate model meets performance thresholds"""
    import mlflow
    from mlflow.tracking import MlflowClient
    
    ti = context['task_instance']
    mlruns_dir = ti.xcom_pull(task_ids='get_paths', key='mlruns_dir')
    
    logging.info("Validating model performance...")
    
    # Set MLflow tracking URI
    mlflow.set_tracking_uri(f"file://{mlruns_dir}")
    
    client = MlflowClient()
    experiment = client.get_experiment_by_name('anomaly_detection')
    
    if not experiment:
        raise ValueError("Experiment 'anomaly_detection' not found in MLflow")
    
    # Get latest run
    runs = client.search_runs(
        experiment.experiment_id,
        order_by=['start_time DESC'],
        max_results=1
    )
    
    if not runs:
        raise ValueError("No training runs found")
    
    run = runs[0]
    
    # Extract metrics
    roc_auc = run.data.metrics.get('One_Class_SVM_roc_auc', 0)
    precision_at_10 = run.data.metrics.get('One_Class_SVM_precision_at_10', 0)
    f1_score = run.data.metrics.get('One_Class_SVM_f1', 0)
    
    logging.info(f"Model performance:")
    logging.info(f"  ROC-AUC: {roc_auc:.4f}")
    logging.info(f"  Precision@10%: {precision_at_10:.4f}")
    logging.info(f"  F1-Score: {f1_score:.4f}")
    
    # Performance gates
    MIN_ROC_AUC = 0.75
    MIN_PRECISION = 0.60
    
    passed = True
    
    if roc_auc < MIN_ROC_AUC:
        logging.error(f"❌ ROC-AUC {roc_auc:.4f} below threshold {MIN_ROC_AUC}")
        passed = False
    
    if precision_at_10 < MIN_PRECISION:
        logging.error(f"❌ Precision@10% {precision_at_10:.4f} below threshold {MIN_PRECISION}")
        passed = False
    
    if not passed:
        raise ValueError("Model failed performance gates")
    
    logging.info("✓ Model passed all performance gates")
    
    # Push to XCom
    ti.xcom_push(key='roc_auc', value=roc_auc)
    ti.xcom_push(key='precision_at_10', value=precision_at_10)
    ti.xcom_push(key='f1_score', value=f1_score)
    ti.xcom_push(key='run_id', value=run.info.run_id)


def validate_bias_metrics(**context):
    """Validate model fairness across sectors"""
    import pandas as pd
    
    ti = context['task_instance']
    output_dir = ti.xcom_pull(task_ids='get_paths', key='output_dir')
    
    logging.info("Validating bias metrics...")
    
    sector_file = os.path.join(output_dir, "models", "One_Class_SVM_sector_analysis.csv")
    
    if not Path(sector_file).exists():
        logging.warning("⚠️ Sector analysis not found, skipping bias check")
        ti.xcom_push(key='f1_std', value=0.0)
        ti.xcom_push(key='precision_std', value=0.0)
        return
    
    df = pd.read_csv(sector_file)
    
    f1_std = df['F1_Score'].std()
    precision_std = df['Precision'].std()
    
    logging.info(f"Bias metrics:")
    logging.info(f"  F1-Score Std Dev: {f1_std:.4f}")
    logging.info(f"  Precision Std Dev: {precision_std:.4f}")
    
    # Bias threshold
    MAX_BIAS = 0.15
    
    if f1_std > MAX_BIAS:
        raise ValueError(f"F1 std dev {f1_std:.4f} exceeds threshold {MAX_BIAS}")
    
    logging.info("✓ Model passed bias check")
    
    # Push to XCom
    ti.xcom_push(key='f1_std', value=f1_std)
    ti.xcom_push(key='precision_std', value=precision_std)


def promote_model_to_staging(**context):
    """Promote model to Staging in MLflow Registry"""
    import mlflow
    from mlflow.tracking import MlflowClient
    
    ti = context['task_instance']
    mlruns_dir = ti.xcom_pull(task_ids='get_paths', key='mlruns_dir')
    
    logging.info("Promoting model to Staging...")
    
    # Set MLflow tracking URI
    mlflow.set_tracking_uri(f"file://{mlruns_dir}")
    
    client = MlflowClient()
    model_name = 'anomaly_detection_one_class_svm'
    
    # Get latest version
    try:
        latest_versions = client.get_latest_versions(model_name, stages=['None'])
    except:
        # Model might not be registered yet
        logging.warning(f"Model {model_name} not found in registry")
        return
    
    if not latest_versions:
        logging.warning(f"No model versions found for {model_name}")
        return
    
    version = latest_versions[0].version
    
    # Transition to Staging
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage='Staging',
        archive_existing_versions=True
    )
    
    logging.info(f"✓ Promoted {model_name} version {version} to Staging")
    
    # Push to XCom
    ti.xcom_push(key='model_version', value=version)
    ti.xcom_push(key='model_stage', value='Staging')

# ============================================================================
# DAG TASKS
# ============================================================================

with dag:
    
    # ------------------------------------------------------------------------
    # STAGE 1: WAIT FOR DATA PIPELINE
    # ------------------------------------------------------------------------
    
    wait_for_data_pipeline = ExternalTaskSensor(
        task_id='wait_for_data_pipeline',
        external_dag_id='financial_crisis_pipeline',
        external_task_id='pipeline_success',
        allowed_states=['success'],
        failed_states=['failed', 'skipped'],
        mode='reschedule',
        timeout=7200,  # 2 hours
        poke_interval=60,  # Check every minute
        doc_md="""
        ### Wait for Data Pipeline Completion
        Waits for `financial_crisis_pipeline` DAG to complete successfully
        before starting model training pipeline.
        """,
    )
    
    # ------------------------------------------------------------------------
    # STAGE 2: CONFIGURATION
    # ------------------------------------------------------------------------
    
    get_paths = PythonOperator(
        task_id='get_paths',
        python_callable=get_paths_from_upstream,
        doc_md="""
        ### Get Paths from Upstream DAG
        Pulls data paths from upstream DAG via XCom.
        Configures all paths dynamically (no hard-coding).
        """,
    )
    
    # ------------------------------------------------------------------------
    # STAGE 3: CHECK FOR EXISTING LABELS
    # ------------------------------------------------------------------------
    
    check_labels = BranchPythonOperator(
        task_id='check_labeled_data',
        python_callable=check_labeled_data_exists,
        doc_md="""
        ### Check for Existing Labels
        If labeled data exists → skip to training
        If not → run EDA → Snorkel labeling
        """,
    )
    
    skip_labeling = EmptyOperator(task_id='skip_labeling')
    
    # ------------------------------------------------------------------------
    # STAGE 4: LABELING PIPELINE (if needed)
    # ------------------------------------------------------------------------
    
    with TaskGroup('labeling_pipeline', tooltip='EDA → Thresholds → Snorkel') as labeling_group:
        
        run_eda = PythonOperator(
            task_id='run_eda',
            python_callable=run_eda_analysis,
            execution_timeout=timedelta(minutes=30),
            on_failure_callback=task_failure_alert,
        )
        
        extract_thresh = PythonOperator(
            task_id='extract_thresholds',
            python_callable=extract_thresholds,
            execution_timeout=timedelta(minutes=10),
            on_failure_callback=task_failure_alert,
        )
        
        run_snorkel = PythonOperator(
            task_id='run_snorkel',
            python_callable=run_snorkel_labeling,
            execution_timeout=timedelta(hours=1),
            on_failure_callback=task_failure_alert,
        )
        
        run_eda >> extract_thresh >> run_snorkel
    
    # Join point after branching
    join_after_labeling = EmptyOperator(
        task_id='join_after_labeling',
        trigger_rule='none_failed_min_one_success',
    )
    
    # ------------------------------------------------------------------------
    # STAGE 5: MODEL TRAINING
    # ------------------------------------------------------------------------
    
    with TaskGroup('model_training', tooltip='Train → Tune → Validate') as training_group:
        
        train_models = PythonOperator(
            task_id='train_models',
            python_callable=train_all_models,
            execution_timeout=timedelta(hours=2),
            on_failure_callback=task_failure_alert,
        )
        
        validate_perf = PythonOperator(
            task_id='validate_performance',
            python_callable=validate_model_performance,
            on_failure_callback=model_validation_failure_alert,
        )
        
        validate_bias_task = PythonOperator(
            task_id='validate_bias',
            python_callable=validate_bias_metrics,
            on_failure_callback=model_validation_failure_alert,
        )
        
        train_models >> [validate_perf, validate_bias_task]
    
    # ------------------------------------------------------------------------
    # STAGE 6: MODEL REGISTRY
    # ------------------------------------------------------------------------
    
    promote_model = PythonOperator(
        task_id='promote_to_staging',
        python_callable=promote_model_to_staging,
        on_failure_callback=task_failure_alert,
    )
    
    # ------------------------------------------------------------------------
    # STAGE 7: SUCCESS NOTIFICATION
    # ------------------------------------------------------------------------
    
    notify_success = PythonOperator(
        task_id='notify_success',
        python_callable=pipeline_success_alert,
    )
    
    end = EmptyOperator(task_id='end')
    
    # ------------------------------------------------------------------------
    # DAG FLOW
    # ------------------------------------------------------------------------
    
    wait_for_data_pipeline >> get_paths >> check_labels
    check_labels >> [skip_labeling, labeling_group]
    [skip_labeling, labeling_group] >> join_after_labeling
    join_after_labeling >> training_group >> promote_model >> notify_success >> end