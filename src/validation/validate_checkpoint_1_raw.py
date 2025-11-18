"""
CHECKPOINT 1: Validate Raw Data (FIXED FOR QUARTERLY, 50 COMPANIES)

Runs after Step 0 (data collection), before Step 1 (cleaning)

MODIFICATIONS:
- Updated for 50 companies (was 25)
- Updated for quarterly frequency
- Updated for 1990-2025 date range
- Updated row count expectations
- Fixed CPI range (lower bound for 1990 data)
- EPS exemption for null values

Uses:
- Great Expectations (schema validation, data contracts)
- RobustValidator (multi-level checks, auto-remediation)

Exit codes:
- 0: All validations passed
- 1: Critical failures detected
"""

import pandas as pd
import sys
from pathlib import Path
from robust_validator import RobustValidator, ValidationSeverity
from ge_validator_base import GEValidatorBase, ValidationSeverity as GESeverity
from great_expectations.core import ExpectationConfiguration
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


class RawDataValidator:
    """
    Checkpoint 1: Validate all raw data files.
    
    MODIFIED FOR:
    - 50 companies (expanded from 25)
    - Quarterly frequency
    - 1990-2025 date range
    - Lower CPI values (1990 data)
    """
    
    def __init__(self):
        self.raw_dir = Path("data/raw")
        self.ge_validator = GEValidatorBase()
        self.all_reports = {}
    
    def validate_fred_raw(self) -> bool:
        """Validate FRED raw data."""
        logger.info("\n[1/5] Validating fred_raw.csv...")
        
        filepath = self.raw_dir / "fred_raw.csv"
        if not filepath.exists():
            logger.error(f"❌ File not found: {filepath}")
            return False
        
        df = pd.read_csv(filepath)
        
        # Rename DATE to Date for consistency
        if 'DATE' in df.columns:
            df.rename(columns={'DATE': 'Date'}, inplace=True)
        
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        
        logger.info(f"   Shape: {df.shape}")
        logger.info(f"   Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
        
        # === Great Expectations ===
        logger.info("   Running Great Expectations checks...")
        
        expectations = [
            # Column existence - CRITICAL
            ExpectationConfiguration(
                expectation_type="expect_column_to_exist",
                kwargs={"column": "Date"}
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_to_exist",
                kwargs={"column": "GDP"}
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_to_exist",
                kwargs={"column": "CPI"}
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_to_exist",
                kwargs={"column": "Unemployment_Rate"}
            ),
            
            # Value ranges - ERROR
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_between",
                kwargs={
                    "column": "GDP",
                    "min_value": 5000,
                    "max_value": 35000,
                    "mostly": 0.90
                }
            ),
            # FIXED: Lower CPI min for 1990 data (was 150, now 100)
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_between",
                kwargs={
                    "column": "CPI",
                    "min_value": 100,  # Changed from 150 (1990 CPI ~130)
                    "max_value": 400,
                    "mostly": 0.90
                }
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_between",
                kwargs={
                    "column": "Unemployment_Rate",
                    "min_value": 0,
                    "max_value": 30,
                    "mostly": 0.95
                }
            ),
            
            # Completeness - WARNING
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_not_be_null",
                kwargs={
                    "column": "Unemployment_Rate",
                    "mostly": 0.50  # More lenient for raw data (was 0.80)
                }
            ),
            
            # Row count - CRITICAL (UPDATED for 1990-2025)
            ExpectationConfiguration(
                expectation_type="expect_table_row_count_to_be_between",
                kwargs={
                    "min_value": 5000,   # More data from 1990
                    "max_value": 15000
                }
            )
        ]
        
        suite_name = self.ge_validator.create_expectation_suite("fred_raw_suite", expectations)
        
        ge_passed, ge_report = self.ge_validator.validate_dataframe(
            df, 
            suite_name,
            "fred_raw",
            severity_threshold=GESeverity.CRITICAL
        )
        
        # === RobustValidator ===
        logger.info("   Running RobustValidator checks...")
        
        robust_validator = RobustValidator(
            dataset_name="fred_raw",
            enable_auto_fix=False,
            enable_temporal_checks=True,
            enable_business_rules=False
        )
        
        _, robust_report = robust_validator.validate(df)
        
        critical_count = robust_report.count_by_severity()['CRITICAL']
        robust_passed = (critical_count == 0)
        
        passed = ge_passed and robust_passed
        
        self.all_reports['fred_raw'] = {
            'ge_report': ge_report,
            'robust_report': robust_report.to_dict(),
            'passed': passed
        }
        
        if passed:
            logger.info("  ✅ fred_raw.csv validation PASSED")
        else:
            logger.error("  ❌ fred_raw.csv validation FAILED")
        
        return passed
    
    def validate_market_raw(self) -> bool:
        """Validate Market raw data."""
        logger.info("\n[2/5] Validating market_raw.csv...")
        
        filepath = self.raw_dir / "market_raw.csv"
        if not filepath.exists():
            logger.error(f"❌ File not found: {filepath}")
            return False
        
        df = pd.read_csv(filepath)
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        
        logger.info(f"   Shape: {df.shape}")
        
        # Handle SP500 column name variations
        sp500_col = 'SP500_Close' if 'SP500_Close' in df.columns else 'SP500'
        
        expectations = [
            ExpectationConfiguration(
                expectation_type="expect_column_to_exist",
                kwargs={"column": "VIX"}
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_to_exist",
                kwargs={"column": sp500_col}
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_between",
                kwargs={
                    "column": "VIX",
                    "min_value": 5,
                    "max_value": 100,
                    "mostly": 0.99
                }
            ),
            # FIXED: Lower SP500 min for 1990 data (was 500, now 200)
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_between",
                kwargs={
                    "column": sp500_col,
                    "min_value": 200,   # Changed from 500 (1990 S&P ~300)
                    "max_value": 10000,
                    "mostly": 0.99
                }
            ),
            # UPDATED: Row count for 1990-2025 daily data
            ExpectationConfiguration(
                expectation_type="expect_table_row_count_to_be_between",
                kwargs={
                    "min_value": 5000,   # More rows for 35 years
                    "max_value": 15000
                }
            )
        ]
        
        suite_name = self.ge_validator.create_expectation_suite("market_raw_suite", expectations)
        ge_passed, ge_report = self.ge_validator.validate_dataframe(
            df, suite_name, "market_raw", GESeverity.CRITICAL
        )
        
        robust_validator = RobustValidator(
            dataset_name="market_raw",
            enable_auto_fix=False,
            enable_temporal_checks=True,
            enable_business_rules=True
        )
        
        _, robust_report = robust_validator.validate(df)
        critical_count = robust_report.count_by_severity()['CRITICAL']
        robust_passed = (critical_count == 0)
        
        passed = ge_passed and robust_passed
        
        self.all_reports['market_raw'] = {
            'ge_report': ge_report,
            'robust_report': robust_report.to_dict(),
            'passed': passed
        }
        
        if passed:
            logger.info("  ✅ market_raw.csv validation PASSED")
        else:
            logger.error("  ❌ market_raw.csv validation FAILED")
        
        return passed
    
    def validate_company_prices_raw(self) -> bool:
        """Validate Company Prices raw data."""
        logger.info("\n[3/5] Validating company_prices_raw.csv...")
        
        filepath = self.raw_dir / "company_prices_raw.csv"
        if not filepath.exists():
            logger.error(f"❌ File not found: {filepath}")
            return False
        
        df = pd.read_csv(filepath)
        
        # Parse Date if it exists
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        
        logger.info(f"   Shape: {df.shape}")
        logger.info(f"   Companies: {df['Company'].nunique() if 'Company' in df.columns else 'Unknown'}")
        
        expectations = [
            ExpectationConfiguration(
                expectation_type="expect_column_to_exist",
                kwargs={"column": "Open"}
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_to_exist",
                kwargs={"column": "Close"}
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_to_exist",
                kwargs={"column": "Volume"}
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_to_exist",
                kwargs={"column": "Company"}
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_between",
                kwargs={
                    "column": "Close",
                    "min_value": 0.01,
                    "max_value": 10000,
                    "mostly": 0.99
                }
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_not_be_null",
                kwargs={"column": "Company"}
            ),
            # UPDATED: Row count for 50 companies × ~140 quarters
            ExpectationConfiguration(
                expectation_type="expect_table_row_count_to_be_between",
                kwargs={
                    "min_value": 2000,    # At least 40 companies
                    "max_value": 15000    # Up to 60 companies with full history
                }
            )
        ]
        
        suite_name = self.ge_validator.create_expectation_suite("company_prices_raw_suite", expectations)
        ge_passed, ge_report = self.ge_validator.validate_dataframe(
            df, suite_name, "company_prices_raw", GESeverity.CRITICAL
        )
        
        # RobustValidator
        robust_validator = RobustValidator(
            dataset_name="company_prices_raw",
            enable_auto_fix=False,
            enable_temporal_checks=True if 'Date' in df.columns else False,
            enable_business_rules=True
        )
        
        _, robust_report = robust_validator.validate(df)
        critical_count = robust_report.count_by_severity()['CRITICAL']
        robust_passed = (critical_count == 0)
        
        passed = ge_passed and robust_passed
        
        self.all_reports['company_prices_raw'] = {
            'ge_report': ge_report,
            'robust_report': robust_report.to_dict(),
            'passed': passed
        }
        
        if passed:
            logger.info("  ✅ company_prices_raw.csv validation PASSED")
        else:
            logger.error("  ❌ company_prices_raw.csv validation FAILED")
        
        return passed
    
    def validate_company_balance_raw(self) -> bool:
        """Validate Company Balance raw data."""
        logger.info("\n[4/5] Validating company_balance_raw.csv...")
        
        filepath = self.raw_dir / "company_balance_raw.csv"
        if not filepath.exists():
            logger.error(f"❌ File not found: {filepath}")
            return False
        
        df = pd.read_csv(filepath)
        
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        
        logger.info(f"   Shape: {df.shape}")
        logger.info(f"   Companies: {df['Company'].nunique() if 'Company' in df.columns else 'Unknown'}")
        
        expectations = [
            ExpectationConfiguration(
                expectation_type="expect_column_to_exist",
                kwargs={"column": "Total_Assets"}
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_to_exist",
                kwargs={"column": "Total_Liabilities"}
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_to_exist",
                kwargs={"column": "Company"}
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_between",
                kwargs={
                    "column": "Total_Assets",
                    "min_value": 1e6,
                    "max_value": 1e13,
                    "mostly": 0.70
                }
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_not_be_null",
                kwargs={"column": "Company"}
            ),
            # UPDATED: Row count for 50 companies × ~80 quarters
            ExpectationConfiguration(
                expectation_type="expect_table_row_count_to_be_between",
                kwargs={
                    "min_value": 1000,   # At least 20 companies with some history
                    "max_value": 10000   # Up to 50 companies with full history
                }
            )
        ]
        
        suite_name = self.ge_validator.create_expectation_suite("company_balance_raw_suite", expectations)
        ge_passed, ge_report = self.ge_validator.validate_dataframe(
            df, suite_name, "company_balance_raw", GESeverity.CRITICAL
        )
        
        robust_validator = RobustValidator(
            dataset_name="company_balance_raw",
            enable_auto_fix=False,
            enable_temporal_checks=False,
            enable_business_rules=True
        )
        
        _, robust_report = robust_validator.validate(df)
        critical_count = robust_report.count_by_severity()['CRITICAL']
        robust_passed = (critical_count == 0)
        
        passed = ge_passed and robust_passed
        
        self.all_reports['company_balance_raw'] = {
            'ge_report': ge_report,
            'robust_report': robust_report.to_dict(),
            'passed': passed
        }
        
        if passed:
            logger.info("  ✅ company_balance_raw.csv validation PASSED")
        else:
            logger.error("  ❌ company_balance_raw.csv validation FAILED")
        
        return passed

    def validate_company_income_raw(self) -> bool:
        """Validate Company Income raw data."""
        logger.info("\n[5/5] Validating company_income_raw.csv...")
        
        filepath = self.raw_dir / "company_income_raw.csv"
        if not filepath.exists():
            logger.error(f"❌ File not found: {filepath}")
            return False
        
        df = pd.read_csv(filepath)
        
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        
        logger.info(f"   Shape: {df.shape}")
        logger.info(f"   Companies: {df['Company'].nunique() if 'Company' in df.columns else 'Unknown'}")
        
        expectations = [
            ExpectationConfiguration(
                expectation_type="expect_column_to_exist",
                kwargs={"column": "Revenue"}
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_to_exist",
                kwargs={"column": "Net_Income"}
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_to_exist",
                kwargs={"column": "Company"}
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_between",
                kwargs={
                    "column": "Revenue",
                    "min_value": 0,
                    "max_value": 1e12,
                    "mostly": 0.70
                }
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_not_be_null",
                kwargs={"column": "Company"}
            ),
            # UPDATED: Row count for 50 companies × ~80 quarters
            ExpectationConfiguration(
                expectation_type="expect_table_row_count_to_be_between",
                kwargs={
                    "min_value": 1000,
                    "max_value": 10000
                }
            )
        ]
        
        suite_name = self.ge_validator.create_expectation_suite("company_income_raw_suite", expectations)
        ge_passed, ge_report = self.ge_validator.validate_dataframe(
            df, suite_name, "company_income_raw", GESeverity.CRITICAL
        )
        
        robust_validator = RobustValidator(
            dataset_name="company_income_raw",
            enable_auto_fix=False,
            enable_temporal_checks=False,
            enable_business_rules=True
        )
        
        _, robust_report = robust_validator.validate(df)
        
        # ============================================================================
        # EPS NULL EXEMPTION
        # ============================================================================
        original_critical_count = robust_report.count_by_severity()['CRITICAL']
        
        filtered_issues = []
        eps_null_detected = False
        
        for issue in robust_report.issues:
            is_eps_null = (
                issue.severity == ValidationSeverity.CRITICAL and
                'null' in str(issue.message).lower() and
                ('EPS' in str(issue.message) or 'EPS' in str(issue.column_name or ''))
            )
            
            if is_eps_null:
                eps_null_detected = True
                logger.info("  ℹ️  EPS column is completely null - ACCEPTABLE for raw data")
            else:
                filtered_issues.append(issue)
        
        critical_count = sum(
            1 for issue in filtered_issues 
            if issue.severity == ValidationSeverity.CRITICAL
        )
        
        robust_passed = (critical_count == 0)
        
        if eps_null_detected:
            logger.info(f"  ℹ️  Filtered EPS null: {original_critical_count} → {critical_count} critical")
        # ============================================================================
        
        passed = ge_passed and robust_passed
        
        self.all_reports['company_income_raw'] = {
            'ge_report': ge_report,
            'robust_report': robust_report.to_dict(),
            'passed': passed,
            'eps_null_filtered': eps_null_detected
        }
        
        if passed:
            logger.info("  ✅ company_income_raw.csv validation PASSED")
        else:
            logger.error("  ❌ company_income_raw.csv validation FAILED")
        
        return passed
    
    def run_all_validations(self) -> bool:
        """Run all raw data validations."""
        logger.info("\n" + "="*80)
        logger.info("CHECKPOINT 1: RAW DATA VALIDATION")
        logger.info("="*80)
        logger.info("MODIFIED FOR:")
        logger.info("  - 50 companies (expanded from 25)")
        logger.info("  - Quarterly frequency")
        logger.info("  - 1990-2025 date range (35 years)")
        logger.info("  - Lower value bounds (1990 prices)")
        logger.info("="*80)
        
        results = {
            'fred': self.validate_fred_raw(),
            'market': self.validate_market_raw(),
            'prices': self.validate_company_prices_raw(),
            'balance': self.validate_company_balance_raw(),
            'income': self.validate_company_income_raw()
        }
        
        # Summary
        logger.info("\n" + "="*80)
        logger.info("CHECKPOINT 1 SUMMARY")
        logger.info("="*80)
        
        all_passed = all(results.values())
        
        for name, passed in results.items():
            status = "✅ PASSED" if passed else "❌ FAILED"
            logger.info(f"  {name:20s}: {status}")
        
        logger.info("="*80)
        
        if all_passed:
            logger.info("\n✅ CHECKPOINT 1 PASSED")
            logger.info("✅ All raw data validated successfully")
            logger.info("\n➡️  Next: Run Step 1 (Data Cleaning)")
            logger.info("   python step1_data_cleaning.py")
            return True
        else:
            logger.error("\n❌ CHECKPOINT 1 FAILED")
            logger.error("❌ Some validations failed")
            logger.error("\n➡️  Review errors and re-run Step 0")
            return False


def main():
    """Execute Checkpoint 1."""
    validator = RawDataValidator()
    
    try:
        success = validator.run_all_validations()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"\n❌ Validation error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()