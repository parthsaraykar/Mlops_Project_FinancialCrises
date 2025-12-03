#!/bin/bash
################################################################################
# Financial Stress Test - CI/CD + Airflow Setup Script
# Sets up complete automation pipeline in 5 minutes
################################################################################

set -e  # Exit on error

echo "🚀 Setting up CI/CD + Airflow Pipeline"
echo "======================================"
echo ""

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Helper functions
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check if running from project root
if [ ! -f "requirements.txt" ]; then
    print_error "Please run this script from project root directory"
    exit 1
fi

echo "📋 Step 1: Create Directory Structure"
echo "--------------------------------------"
mkdir -p .github/workflows
mkdir -p dags
mkdir -p logs
mkdir -p plugins
print_success "Directories created"
echo ""

echo "📄 Step 2: Setup CI/CD Workflow"
echo "-------------------------------"
if [ -f ".github/workflows/ml_pipeline_ci_cd.yml" ]; then
    print_success "GitHub Actions workflow found"
else
    print_warning "GitHub Actions workflow not found - create manually from artifact"
fi
echo ""

echo "🌊 Step 3: Setup Airflow"
echo "-----------------------"

# Check if Airflow is installed
if ! command -v airflow &> /dev/null; then
    echo "Installing Apache Airflow..."
    pip install apache-airflow==2.7.0 --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-2.7.0/constraints-3.11.txt"
    print_success "Airflow installed"
else
    print_success "Airflow already installed"
fi

# Set Airflow home
export AIRFLOW_HOME=~/airflow
echo "export AIRFLOW_HOME=~/airflow" >> ~/.bashrc
print_success "AIRFLOW_HOME set to ~/airflow"

# Initialize Airflow database
if [ ! -f "$AIRFLOW_HOME/airflow.db" ]; then
    echo "Initializing Airflow database..."
    airflow db init
    print_success "Airflow database initialized"
else
    print_success "Airflow database already exists"
fi

# Create Airflow directories
mkdir -p $AIRFLOW_HOME/dags
mkdir -p $AIRFLOW_HOME/logs
mkdir -p $AIRFLOW_HOME/plugins
print_success "Airflow directories created"

# Copy DAGs to Airflow
if [ -f "dags/financial_crisis_pipeline.py" ]; then
    cp dags/financial_crisis_pipeline.py $AIRFLOW_HOME/dags/
    print_success "Data Pipeline DAG copied to Airflow"
else
    print_warning "Data Pipeline DAG not found"
fi

if [ -f "dags/financial_stress_model3_pipeline.py" ]; then
    cp dags/financial_stress_model3_pipeline.py $AIRFLOW_HOME/dags/
    print_success "Model Pipeline DAG copied to Airflow"
else
    print_warning "Model Pipeline DAG not found"
fi

# Create Airflow admin user (only if doesn't exist)
if ! airflow users list 2>/dev/null | grep -q "admin"; then
    echo "Creating Airflow admin user..."
    airflow users create \
        --username admin \
        --firstname Admin \
        --lastname User \
        --role Admin \
        --email admin@example.com \
        --password admin
    print_success "Admin user created (username: admin, password: admin)"
else
    print_success "Admin user already exists"
fi
echo ""

echo "🐳 Step 4: Setup Docker Compose for Airflow (Optional)"
echo "------------------------------------------------------"
cat > docker-compose.airflow.yml << 'EOF'
version: '3.8'

services:
  postgres:
    image: postgres:13
    environment:
      POSTGRES_USER: airflow
      POSTGRES_PASSWORD: airflow
      POSTGRES_DB: airflow
    volumes:
      - postgres-db-volume:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  airflow-webserver:
    image: apache/airflow:2.7.0-python3.11
    command: webserver
    depends_on:
      - postgres
    environment:
      AIRFLOW__CORE__EXECUTOR: LocalExecutor
      AIRFLOW__DATABASE__SQL_ALCHEMY_CONN: postgresql+psycopg2://airflow:airflow@postgres/airflow
      AIRFLOW__CORE__FERNET_KEY: ''
      AIRFLOW__CORE__DAGS_ARE_PAUSED_AT_CREATION: 'true'
      AIRFLOW__CORE__LOAD_EXAMPLES: 'false'
      AIRFLOW__WEBSERVER__EXPOSE_CONFIG: 'true'
      PROJECT_ROOT: /opt/airflow/project
    volumes:
      - ./dags:/opt/airflow/dags
      - ./logs:/opt/airflow/logs
      - ./plugins:/opt/airflow/plugins
      - ./src:/opt/airflow/project/src
      - ./configs:/opt/airflow/project/configs
      - ./outputs:/opt/airflow/project/outputs
      - ./data:/opt/airflow/project/data
      - ./models:/opt/airflow/project/models
      - ./mlruns:/opt/airflow/project/mlruns
    ports:
      - "8080:8080"

  airflow-scheduler:
    image: apache/airflow:2.7.0-python3.11
    command: scheduler
    depends_on:
      - postgres
    environment:
      AIRFLOW__CORE__EXECUTOR: LocalExecutor
      AIRFLOW__DATABASE__SQL_ALCHEMY_CONN: postgresql+psycopg2://airflow:airflow@postgres/airflow
      AIRFLOW__CORE__FERNET_KEY: ''
      PROJECT_ROOT: /opt/airflow/project
    volumes:
      - ./dags:/opt/airflow/dags
      - ./logs:/opt/airflow/logs
      - ./plugins:/opt/airflow/plugins
      - ./src:/opt/airflow/project/src
      - ./configs:/opt/airflow/project/configs
      - ./outputs:/opt/airflow/project/outputs
      - ./data:/opt/airflow/project/data
      - ./models:/opt/airflow/project/models
      - ./mlruns:/opt/airflow/project/mlruns

volumes:
  postgres-db-volume:
EOF
print_success "Docker Compose file created (docker-compose.airflow.yml)"
echo ""

echo "📝 Step 5: Create Helper Scripts"
echo "--------------------------------"

# Create start script
cat > start_airflow.sh << 'EOF'
#!/bin/bash
echo "Starting Airflow..."
export AIRFLOW_HOME=~/airflow

# Start webserver in background
airflow webserver --port 8080 > logs/webserver.log 2>&1 &
WEBSERVER_PID=$!
echo "Webserver started (PID: $WEBSERVER_PID)"

# Start scheduler in background
airflow scheduler > logs/scheduler.log 2>&1 &
SCHEDULER_PID=$!
echo "Scheduler started (PID: $SCHEDULER_PID)"

echo ""
echo "✓ Airflow is running!"
echo "  Webserver: http://localhost:8080"
echo "  Username: admin"
echo "  Password: admin"
echo ""
echo "To stop: ./stop_airflow.sh"
EOF
chmod +x start_airflow.sh
print_success "Created start_airflow.sh"

# Create stop script
cat > stop_airflow.sh << 'EOF'
#!/bin/bash
echo "Stopping Airflow..."
pkill -f "airflow webserver"
pkill -f "airflow scheduler"
echo "✓ Airflow stopped"
EOF
chmod +x stop_airflow.sh
print_success "Created stop_airflow.sh"

# Create test script
cat > test_dag.sh << 'EOF'
#!/bin/bash
export AIRFLOW_HOME=~/airflow
DAG_ID="financial_stress_model3_complete_pipeline"

echo "Testing DAG: $DAG_ID"
echo "===================="
echo ""

echo "1. Listing all tasks..."
airflow tasks list $DAG_ID --tree
echo ""

echo "2. Testing validate_data task..."
airflow tasks test $DAG_ID validate_data 2025-01-01
echo ""

echo "✓ Test complete!"
echo ""
echo "To run full DAG:"
echo "  airflow dags trigger $DAG_ID"
EOF
chmod +x test_dag.sh
print_success "Created test_dag.sh"

echo ""

echo "✅ Setup Complete!"
echo "=================="
echo ""
echo "📋 Next Steps:"
echo ""
echo "1. Set PROJECT_ROOT environment variable:"
echo "   export PROJECT_ROOT=$(pwd)"
echo "   echo 'export PROJECT_ROOT=$(pwd)' >> ~/.bashrc"
echo ""
echo "2. Import Airflow variables:"
echo "   airflow variables set project_root $(pwd)"
echo "   airflow variables set FRED_API_KEY 'your_key_here'"
echo ""
echo "3. Start Airflow:"
echo "   ./start_airflow.sh"
echo "   # Or: make airflow-start"
echo ""
echo "4. Open UI:"
echo "   http://localhost:8080 (admin/admin)"
echo ""
echo "5. Enable DAGs in UI:"
echo "   - financial_crisis_pipeline (enable this)"
echo "   - financial_stress_model3_pipeline (leave off - auto-triggered)"
echo ""
echo "6. Trigger pipeline:"
echo "   make pipeline-full"
echo "   # Or click 'Play' button in Airflow UI"
echo ""
echo "📊 Monitoring:"
echo "   - Airflow UI: http://localhost:8080"
echo "   - MLflow UI: make mlflow (http://localhost:5000)"
echo ""
echo "🎯 You're ready to go!"
echo ""
