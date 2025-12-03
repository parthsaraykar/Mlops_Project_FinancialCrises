"""
Financial Crisis Detection Pipeline - Clean & Modular DAG with Alerting
========================================================================
Includes all validation checkpoints + triggers Model 3 Pipeline on success
 
Author: MLOps Group11 Team
"""
 
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.utils.dates import days_ago
from datetime import timedelta
import os
import sys
 
# Get project directory from environment or Airflow Variable
PROJECT_DIR = os.getenv('PROJECT_ROOT', '/opt/airflow/project')
sys.path.insert(0, PROJECT_DIR)
 
# Import your existing alerting system
try:
    from src.monitoring.alerting import AlertManager
    ALERTING_AVAILABLE = True
except ImportError:
    print("WARNING: AlertManager not available. Alerts disabled.")
    ALERTING_AVAILABLE = False
 
# ==============================================================================
# CONFIGURATION
# ==============================================================================
 
# Pipeline steps configuration
PIPELINE_STEPS = [
    ('step0_collect_data', 'src/data/step0_data_collection.py', 'Collect data from APIs', 90, True),
    ('validate_checkpoint_1', 'src/validation/validate_checkpoint_1_raw.py', 'Validate raw data', 10, True),
    ('step1_data_cleaning_and_merging', 'src/data/step1_data_cleaning_and_merging.py', 'Clean data (PIT correct)', 20, True),
    ('validate_checkpoint_2', 'src/validation/validate_checkpoint_2_clean.py', 'Validate clean data', 10, True),
    ('step2_feature_engineering', 'src/data/step2_feature_engineering.py', 'Engineer features', 20, False),
    ('step3_bias_detection_with_explicit_slicing', 'src/data/step3_bias_detection_with_explicit_slicing.py', 'Detect bias', 10, False),
    ('step4_anomaly_detection', 'src/data/step4_anomaly_detection.py', 'Detect anomalies', 10, False),
    ('step5_drift_detection', 'src/data/step5_drift_detection.py', 'Detect drift', 10, False),
]
 
# ==============================================================================
# ALERTING CALLBACKS USING YOUR EXISTING SYSTEM
# ==============================================================================
 
def task_failure_alert(context):
    """Send alert on task failure using your AlertManager"""
    if not ALERTING_AVAILABLE:
        return
    try:
        task = context.get('task_instance')
        dag_run = context.get('dag_run')
        exception = context.get('exception')
        
        alert_manager = AlertManager()
        
        is_validation = 'validate' in task.task_id
        is_critical = any(step[0] == task.task_id and step[4] for step in PIPELINE_STEPS)
        severity = 'CRITICAL' if (is_validation or is_critical) else 'ERROR'
        
        message = f"""
        Pipeline Task Failed: {task.task_id}
        DAG: {task.dag_id}
        Execution Date: {dag_run.execution_date}
        Task: {task.task_id}
        Error: {str(exception) if exception else 'Check logs for details'}
        Log URL: {task.log_url}
        """
        
        if is_validation:
            message += "\nWARNING: Data Validation Failed - Pipeline stopped to prevent bad data propagation."
        
        alert_manager.send_alert(
            message=message,
            severity=severity,
            component=task.task_id,
            alert_type='PIPELINE_FAILURE'
        )
        print(f"Alert sent for {task.task_id} failure")
    except Exception as e:
        print(f"Failed to send alert: {str(e)}")
 
 
def pipeline_success_alert(**context):
    """Send alert on pipeline success using your AlertManager"""
    if not ALERTING_AVAILABLE:
        return
    try:
        dag_run = context.get('dag_run')
        duration = dag_run.end_date - dag_run.start_date if dag_run.end_date else "N/A"
        
        alert_manager = AlertManager()
        
        message = f"""
        SUCCESS: Financial Crisis Pipeline Completed
        Execution Date: {dag_run.execution_date}
        Duration: {duration}
        
        Pipeline Summary:
        - Data collected & validated (Checkpoint 1)
        - Data cleaned & validated (Checkpoint 2)
        - Features engineered
        - Bias detection completed
        - Anomaly detection completed
        - Drift detection completed
        
        Data ready for model training!
        Triggering Model 3 Pipeline automatically...
        """
        
        alert_manager.send_alert(
            message=message,
            severity='INFO',
            component='pipeline',
            alert_type='PIPELINE_SUCCESS'
        )
        print("Success alert sent")
    except Exception as e:
        print(f"Failed to send success alert: {str(e)}")
 
 
def validation_failure_alert(context):
    """Special alert for validation checkpoint failures"""
    if not ALERTING_AVAILABLE:
        return
    try:
        task = context.get('task_instance')
        alert_manager = AlertManager()
        
        checkpoint_num = '1' if 'checkpoint_1' in task.task_id else \
                        '2' if 'checkpoint_2' in task.task_id else '3'
        
        message = f"""
        CRITICAL: Validation Checkpoint {checkpoint_num} Failed
        Task: {task.task_id}
        Execution Date: {context.get('dag_run').execution_date}
        
        Data quality issues detected. Pipeline has been stopped.
        
        Action Required:
        1. Check validation report: data/validation_reports/
        2. Review failed expectations in Great Expectations
        3. Fix data quality issues
        4. Re-run pipeline
        
        Log URL: {task.log_url}
        """
        
        alert_manager.send_alert(
            message=message,
            severity='CRITICAL',
            component=f'validation_checkpoint_{checkpoint_num}',
            alert_type='VALIDATION_FAILURE'
        )
        print(f"CRITICAL validation alert sent for checkpoint {checkpoint_num}")
    except Exception as e:
        print(f"Failed to send validation alert: {str(e)}")


def push_data_paths(**context):
    """
    Push final data paths to XCom for downstream Model 3 pipeline
    """
    ti = context['task_instance']
    
    # Build all data paths
    data_paths = {
        'project_dir': PROJECT_DIR,
        'merged_data_path': os.path.join(PROJECT_DIR, 'data', 'processed', 'merged_quarterly_data.csv'),
        'raw_data_dir': os.path.join(PROJECT_DIR, 'data', 'raw'),
        'processed_data_dir': os.path.join(PROJECT_DIR, 'data', 'processed'),
        'validation_reports_dir': os.path.join(PROJECT_DIR, 'data', 'validation_reports'),
    }
    
    # Push all paths to XCom
    for key, value in data_paths.items():
        ti.xcom_push(key=key, value=value)
    
    import logging
    logging.info("✓ Data paths pushed to XCom for Model 3 pipeline")
    for key, value in data_paths.items():
        logging.info(f"  {key}: {value}")
 
 
# ==============================================================================
# DAG DEFAULT ARGS
# ==============================================================================
 
default_args = {
    'owner': 'mlops-team',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'on_failure_callback': task_failure_alert,
}
 
# ==============================================================================
# DAG DEFINITION
# ==============================================================================
 
with DAG(
    'financial_crisis_pipeline',
    default_args=default_args,
    description='Financial Crisis Pipeline with Validation & Custom Alerting',
    schedule_interval='0 1 * * 0',  # Weekly Sunday 1 AM
    start_date=days_ago(1),
    catchup=False,
    tags=['mlops', 'financial', 'validation', 'alerting', 'data-pipeline'],
    max_active_runs=1,
) as dag:
    
    # ==========================================================================
    # CREATE TASKS DYNAMICALLY
    # ==========================================================================
    
    tasks = {}
    
    for task_id, script, desc, timeout, critical in PIPELINE_STEPS:
        # Validation checkpoints get special alerting
        callbacks = {}
        if 'validate' in task_id:
            callbacks['on_failure_callback'] = validation_failure_alert
        else:
            callbacks['on_failure_callback'] = task_failure_alert
        
        tasks[task_id] = BashOperator(
            task_id=task_id,
            bash_command=f"""
            cd {PROJECT_DIR} && \
            echo "{'Validating' if 'validate' in task_id else 'Running'}: {desc}..." && \
            python {script}
            """,
            execution_timeout=timedelta(minutes=timeout),
            **callbacks
        )
    
    # ==========================================================================
    # PUSH DATA PATHS TO XCOM
    # ==========================================================================
    
    push_paths = PythonOperator(
        task_id='push_data_paths',
        python_callable=push_data_paths,
        trigger_rule='all_success'
    )
    
    # ==========================================================================
    # SUCCESS NOTIFICATION TASK
    # ==========================================================================
    
    pipeline_success = PythonOperator(
        task_id='pipeline_success',
        python_callable=pipeline_success_alert,
        trigger_rule='all_success'
    )
    
    # ==========================================================================
    # TRIGGER MODEL 3 PIPELINE
    # ==========================================================================
    
    trigger_model3_pipeline = TriggerDagRunOperator(
        task_id='trigger_model3_pipeline',
        trigger_dag_id='financial_stress_model3_pipeline',
        wait_for_completion=False,  # Don't wait, trigger and continue
        trigger_rule='all_success',
        conf={'source_dag': 'financial_crisis_pipeline'},  # Pass context
    )
    
    # ==========================================================================
    # DEFINE DEPENDENCIES (LINEAR FLOW)
    # ==========================================================================
    
    # Chain all pipeline steps
    for i in range(len(PIPELINE_STEPS) - 1):
        current_task = PIPELINE_STEPS[i][0]
        next_task = PIPELINE_STEPS[i + 1][0]
        tasks[current_task] >> tasks[next_task]
    
    # After all data pipeline steps complete
    tasks[PIPELINE_STEPS[-1][0]] >> push_paths >> pipeline_success >> trigger_model3_pipeline