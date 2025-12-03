.PHONY: help setup install test train docker airflow clean all

.DEFAULT_GOAL := help

# Color output
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[1;33m
RED := \033[0;31m
NC := \033[0m # No Color

# Project configuration
PROJECT_ROOT := $(shell pwd)
AIRFLOW_HOME := $(HOME)/airflow

##@ General

help: ## Display this help message
	@echo "$(BLUE)Financial Stress Test - Complete MLOps System$(NC)"
	@echo "$(BLUE)===============================================$(NC)"
	@echo ""
	@awk 'BEGIN {FS = ":.*##"; printf "Usage:\n  make $(YELLOW)<target>$(NC)\n"} /^[a-zA-Z_0-9-]+:.*?##/ { printf "  $(GREEN)%-25s$(NC) %s\n", $$1, $$2 } /^##@/ { printf "\n$(BLUE)%s$(NC)\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

##@ Setup & Installation

setup: ## Complete setup (Data + Model Pipeline)
	@echo "$(BLUE)Setting up integrated DAG system...$(NC)"
	@bash setup_cicd_airflow.sh
	@make airflow-import-variables
	@echo "$(GREEN)✓ Setup complete!$(NC)"

install: ## Install Python dependencies
	@echo "$(BLUE)Installing dependencies...$(NC)"
	pip install -r requirements.txt
	@echo "$(GREEN)✓ Dependencies installed$(NC)"

airflow-import-variables: ## Import Airflow variables from config
	@echo "$(BLUE)Importing Airflow variables...$(NC)"
	airflow variables import configs/airflow_variables.json || true
	airflow variables set project_root "$(PROJECT_ROOT)"
	@echo "$(GREEN)✓ Variables imported$(NC)"

##@ Integrated Pipeline

pipeline-full: ## Run complete pipeline (Data → Model)
	@echo "$(BLUE)Running complete integrated pipeline...$(NC)"
	@echo "  1. Data Pipeline (2-3 hours)"
	@echo "  2. Model Pipeline (auto-triggered, 45 min)"
	airflow dags trigger financial_crisis_pipeline
	@echo "$(GREEN)✓ Pipeline triggered$(NC)"
	@echo "Monitor: http://localhost:8080"

pipeline-data: ## Run data pipeline only
	@echo "$(BLUE)Running data pipeline...$(NC)"
	airflow dags trigger financial_crisis_pipeline
	@echo "$(GREEN)✓ Data pipeline triggered$(NC)"

pipeline-model: ## Run model pipeline only (manual)
	@echo "$(BLUE)Running model pipeline...$(NC)"
	airflow dags trigger financial_stress_model3_pipeline
	@echo "$(GREEN)✓ Model pipeline triggered$(NC)"

pipeline-status: ## Check pipeline status
	@echo "$(BLUE)Pipeline Status$(NC)"
	@echo "$(BLUE)===============$(NC)"
	@echo ""
	@echo "Data Pipeline:"
	@airflow dags list-runs -d financial_crisis_pipeline --limit 5 || echo "  $(YELLOW)Not run yet$(NC)"
	@echo ""
	@echo "Model Pipeline:"
	@airflow dags list-runs -d financial_stress_model3_pipeline --limit 5 || echo "  $(YELLOW)Not run yet$(NC)"

##@ Airflow Management

airflow-init: ## Initialize Airflow database
	@echo "$(BLUE)Initializing Airflow...$(NC)"
	export AIRFLOW_HOME=$(AIRFLOW_HOME) && airflow db init
	@echo "$(GREEN)✓ Airflow initialized$(NC)"

airflow-create-user: ## Create Airflow admin user
	@echo "$(BLUE)Creating Airflow user...$(NC)"
	export AIRFLOW_HOME=$(AIRFLOW_HOME) && airflow users create \
		--username admin \
		--firstname Admin \
		--lastname User \
		--role Admin \
		--email admin@example.com \
		--password admin
	@echo "$(GREEN)✓ Admin user created (admin/admin)$(NC)"

airflow-copy-dags: ## Copy DAGs to Airflow directory
	@echo "$(BLUE)Copying DAG files...$(NC)"
	mkdir -p $(AIRFLOW_HOME)/dags
	cp dags/financial_crisis_pipeline.py $(AIRFLOW_HOME)/dags/
	cp dags/financial_stress_model3_pipeline.py $(AIRFLOW_HOME)/dags/
	@echo "$(GREEN)✓ DAGs copied$(NC)"
	@ls -la $(AIRFLOW_HOME)/dags/

airflow-start: airflow-copy-dags ## Start Airflow (webserver + scheduler)
	@echo "$(BLUE)Starting Airflow...$(NC)"
	@./start_airflow.sh || (export AIRFLOW_HOME=$(AIRFLOW_HOME) && \
		airflow webserver --port 8080 > logs/webserver.log 2>&1 & \
		airflow scheduler > logs/scheduler.log 2>&1 &)
	@echo "$(GREEN)✓ Airflow started$(NC)"
	@echo "Open: http://localhost:8080 (admin/admin)"

airflow-stop: ## Stop Airflow
	@echo "$(BLUE)Stopping Airflow...$(NC)"
	@./stop_airflow.sh || (pkill -f "airflow webserver" && pkill -f "airflow scheduler")
	@echo "$(GREEN)✓ Airflow stopped$(NC)"

airflow-restart: airflow-stop airflow-start ## Restart Airflow

airflow-logs: ## View Airflow logs
	@tail -100 $(AIRFLOW_HOME)/logs/scheduler/latest/*.log

airflow-test-data: ## Test data pipeline DAG
	@echo "$(BLUE)Testing data pipeline DAG...$(NC)"
	python $(AIRFLOW_HOME)/dags/financial_crisis_pipeline.py
	@echo "$(GREEN)✓ DAG syntax valid$(NC)"

airflow-test-model: ## Test model pipeline DAG
	@echo "$(BLUE)Testing model pipeline DAG...$(NC)"
	python $(AIRFLOW_HOME)/dags/financial_stress_model3_pipeline.py
	@echo "$(GREEN)✓ DAG syntax valid$(NC)"

airflow-test-all: airflow-test-data airflow-test-model ## Test all DAGs

##@ Docker

docker-build: ## Build Docker image
	@echo "$(BLUE)Building Docker image...$(NC)"
	docker build -t financial-stress-model3:latest -f docker/Dockerfile.train .
	@echo "$(GREEN)✓ Docker image built$(NC)"
	@docker images | grep financial-stress

docker-airflow-up: ## Start Airflow with Docker Compose
	@echo "$(BLUE)Starting Airflow with Docker...$(NC)"
	docker-compose -f docker-compose.airflow.yml up -d
	@echo "$(GREEN)✓ Airflow started$(NC)"
	@echo "Open: http://localhost:8080 (admin/admin)"

docker-airflow-down: ## Stop Airflow Docker Compose
	@echo "$(BLUE)Stopping Airflow Docker...$(NC)"
	docker-compose -f docker-compose.airflow.yml down
	@echo "$(GREEN)✓ Airflow stopped$(NC)"

docker-airflow-logs: ## View Airflow Docker logs
	docker-compose -f docker-compose.airflow.yml logs -f

##@ Model Training (Local)

train: ## Train models locally (no Airflow)
	@echo "$(BLUE)Training models locally...$(NC)"
	python src/models/train_anomaly_detection.py
	@echo "$(GREEN)✓ Training complete$(NC)"

eda: ## Run EDA locally
	@echo "$(BLUE)Running EDA...$(NC)"
	python src/eda/eda.py
	@echo "$(GREEN)✓ EDA complete$(NC)"

snorkel: ## Run Snorkel labeling locally
	@echo "$(BLUE)Running Snorkel...$(NC)"
	python src/labeling/auto_threshold_extractor.py
	python src/labeling/snorkel_pipeline.py
	@echo "$(GREEN)✓ Snorkel complete$(NC)"

##@ Monitoring

mlflow: ## Start MLflow UI
	@echo "$(BLUE)Starting MLflow UI...$(NC)"
	@echo "Open: http://localhost:5000"
	mlflow ui --port 5000

status: ## Show complete system status
	@echo "$(BLUE)System Status$(NC)"
	@echo "$(BLUE)═════════════$(NC)"
	@echo ""
	@echo "$(YELLOW)Data Pipeline:$(NC)"
	@[ -f "data/processed/merged_quarterly_data.csv" ] && echo "  $(GREEN)✓$(NC) Merged data" || echo "  $(RED)✗$(NC) Merged data"
	@[ -f "outputs/snorkel/data/snorkel_labeled_only.csv" ] && echo "  $(GREEN)✓$(NC) Labeled data" || echo "  $(RED)✗$(NC) Labeled data"
	@echo ""
	@echo "$(YELLOW)Models:$(NC)"
	@[ -f "models/anomaly_detection/One_Class_SVM/model.pkl" ] && echo "  $(GREEN)✓$(NC) One-Class SVM" || echo "  $(RED)✗$(NC) One-Class SVM"
	@[ -f "models/anomaly_detection/Isolation_Forest/model.pkl" ] && echo "  $(GREEN)✓$(NC) Isolation Forest" || echo "  $(RED)✗$(NC) Isolation Forest"
	@[ -f "models/anomaly_detection/LOF/model.pkl" ] && echo "  $(GREEN)✓$(NC) LOF" || echo "  $(RED)✗$(NC) LOF"
	@echo ""
	@echo "$(YELLOW)Services:$(NC)"
	@pgrep -f "airflow webserver" > /dev/null && echo "  $(GREEN)✓$(NC) Airflow running" || echo "  $(YELLOW)○$(NC) Airflow stopped"
	@pgrep -f "mlflow ui" > /dev/null && echo "  $(GREEN)✓$(NC) MLflow UI running" || echo "  $(YELLOW)○$(NC) MLflow UI stopped"
	@docker ps | grep -q financial-stress && echo "  $(GREEN)✓$(NC) Docker running" || echo "  $(YELLOW)○$(NC) Docker stopped"
	@echo ""
	@echo "$(YELLOW)Recent DAG Runs:$(NC)"
	@airflow dags list-runs --limit 3 2>/dev/null || echo "  $(YELLOW)Airflow not running$(NC)"

logs-data: ## View data pipeline logs
	@echo "$(BLUE)Recent data pipeline logs:$(NC)"
	@tail -50 logs/data_pipeline.log 2>/dev/null || echo "No logs yet"

logs-model: ## View model pipeline logs
	@echo "$(BLUE)Recent model pipeline logs:$(NC)"
	@tail -50 logs/model_pipeline.log 2>/dev/null || echo "No logs yet"

##@ Testing

test: ## Run unit tests
	@echo "$(BLUE)Running tests...$(NC)"
	pytest tests/unit/ -v --cov=src
	@echo "$(GREEN)✓ Tests complete$(NC)"

validate-data: ## Validate data with Great Expectations
	@echo "$(BLUE)Validating data...$(NC)"
	python src/validation/run_validation.py --checkpoint-name quarterly_checkpoint
	@echo "$(GREEN)✓ Validation complete$(NC)"

##@ Cleanup

clean: ## Clean temporary files
	@echo "$(BLUE)Cleaning temporary files...$(NC)"
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".pytest_cache" -delete
	rm -rf .coverage htmlcov/
	@echo "$(GREEN)✓ Cleanup complete$(NC)"

clean-airflow: ## Clean Airflow metadata
	@echo "$(YELLOW)⚠️  This will reset Airflow database!$(NC)"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		rm -rf $(AIRFLOW_HOME)/airflow.db; \
		rm -rf $(AIRFLOW_HOME)/logs/*; \
		echo "$(GREEN)✓ Airflow cleaned$(NC)"; \
	fi

##@ Demo & Presentation

demo: ## Quick demo setup
	@echo "$(BLUE)Setting up demo...$(NC)"
	@make install
	@make airflow-start
	@sleep 5
	@make mlflow &
	@echo ""
	@echo "$(GREEN)✓ Demo ready!$(NC)"
	@echo ""
	@echo "Open these in your browser:"
	@echo "  - Airflow: http://localhost:8080 (admin/admin)"
	@echo "  - MLflow:  http://localhost:5000"
	@echo ""
	@echo "Trigger pipeline:"
	@echo "  make pipeline-full"

demo-stop: ## Stop demo
	@make airflow-stop
	@pkill -f "mlflow ui" || true
	@echo "$(GREEN)✓ Demo stopped$(NC)"

present: ## Prepare for presentation
	@echo "$(BLUE)Preparing for presentation...$(NC)"
	@make status
	@echo ""
	@echo "$(GREEN)✓ System ready for presentation!$(NC)"
	@echo ""
	@echo "Demo script:"
	@echo "  1. Show: make status"
	@echo "  2. Show: Airflow UI (http://localhost:8080)"
	@echo "  3. Trigger: make pipeline-full"
	@echo "  4. Show: MLflow UI (http://localhost:5000)"

##@ Quick Start

all: setup install airflow-start ## Complete setup and start
	@echo "$(GREEN)"
	@echo "╔════════════════════════════════════════╗"
	@echo "║  🎉 System Ready!                     ║"
	@echo "╚════════════════════════════════════════╝"
	@echo "$(NC)"
	@echo "Access:"
	@echo "  - Airflow: http://localhost:8080 (admin/admin)"
	@echo "  - MLflow:  http://localhost:5000"
	@echo ""
	@echo "Trigger pipeline:"
	@echo "  make pipeline-full"
