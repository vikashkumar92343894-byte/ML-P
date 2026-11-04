import sys
import subprocess
from src.utils.logger import get_logger

logger = get_logger("master_pipeline")

def run_step(step_name: str, command: list):
    logger.info(f"==> Running Step: {step_name}...")
    result = subprocess.run(command, check=True)
    if result.returncode != 0:
        logger.error(f"Step {step_name} failed with return code {result.returncode}")
        sys.exit(result.returncode)
    logger.info(f"==> Step {step_name} completed successfully.\n")

def main():
    logger.info("Starting Full End-to-End Pipeline Execution...\n")
    
    # 1. Clean transactions
    run_step("1. Data Cleaning & Unified Schema", [sys.executable, "-m", "src.data.cleaner"])
    
    # 2. Build model features
    run_step("2. Feature Engineering", [sys.executable, "-m", "src.features.build_features"])
    
    # 3. Train recurring model
    run_step("3. Train Recurring Classifier", [sys.executable, "-m", "src.models.train_recurring"])
    
    # 4. Train anomaly model
    run_step("4. Train Anomaly Model", [sys.executable, "-m", "src.models.train_anomaly"])
    
    # 5. Run test suite
    run_step("5. Run Test Suite", [sys.executable, "-m", "pytest", "tests", "-v"])
    
    logger.info("All pipeline stages completed successfully!")

if __name__ == "__main__":
    main()
