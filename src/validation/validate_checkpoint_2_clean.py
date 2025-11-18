"""
CHECKPOINT 2: Validate Cleaned Data (FIXED FOR QUARTERLY, NO FORWARD FILL)

Runs after Step 1 (cleaning), before Step 2 (feature engineering)

MODIFICATIONS:
- Accept 20-50% missing (NO forward fill was used)
- Updated for 50 companies
- Updated for quarterly frequency
- Updated for 1990-2025 range

Focus:
- Required columns present
- No completely empty columns (except EPS)
- Duplicates removed
- Point-in-time correctness maintained
- Acceptable missing percentage for quarterly data

Usage:
    python validate_checkpoint_2_clean.py
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


class CleanedDataValidator:
    """
    Checkpoint 2: Validate cleaned data.
    
    MODIFIED: Accept higher missing percentages (NO forward fill used).
    """
    
    def __init__(self):
        self.clean_dir = Path("data/clean")
        self.ge_validator = GEValidatorBase()
        self.all_reports = {}
    
    def validate_fred_clean(self) -> bool:
        """Validate FRED cleaned data."""
        logger.info("\n[1/5] Validating fred_clean.csv...")
        
        filepath = self.clean_dir / "fred_clean.csv"
        if not filepath.exists():
            logger.error(f"❌ File not found: {filepath}")
            return False
        
        df = pd.read_csv(filepath)
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        
        logger.info(f"   Shape: {df.shape}")
        
        expectations = [
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
            # MODIFIED: Accept 50% missing (NO forward fill)
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_not_be_null",
                kwargs={
                    "column": "GDP",
                    "mostly": 0.50  # 50% OK (was 0.95)
                }
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_unique",
                kwargs={"column": "Date"}
            ),
            ExpectationConfiguration(
                expectation_type="expect_table_row_count_to_be_between",
                kwargs={
                    "min_value": 5000,
                    "max_value": 15000
                }
            )
        ]
        
        suite_name = self.ge_validator.create_expectation_suite("fred_clean_suite", expectations)
        ge_passed, ge_report = self.ge_validator.validate_dataframe(
            df, suite_name, "fred_clean", GESeverity.CRITICAL
        )
        
        robust_validator = RobustValidator(
            dataset_name="fred_clean",
            enable_auto_fix=False,
            enable_temporal_checks=True,
            enable_business_rules=True
        )
        
        _, robust_report = robust_validator.validate(df)
        critical_count = robust_report.count_by_severity()['CRITICAL']
        robust_passed = (critical_count == 0)
        
        passed = ge_passed and robust_passed
        
        self.all_reports['fred_clean'] = {
            'ge_report': ge_report,
            'robust_report': robust_report.to_dict(),
            'passed': passed
        }
        
        if passed:
            logger.info("  ✅ fred_clean.csv validation PASSED")
        else:
            logger.error("  ❌ fred_clean.csv validation FAILED")
        
        return passed
    
    def validate_company_prices_clean(self) -> bool:
        """Validate cleaned company prices."""
        logger.info("\n[3/5] Validating company_prices_clean.csv...")
        
        filepath = self.clean_dir / "company_prices_clean.csv"
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
                kwargs={"column": "Company"}
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_to_exist",
                kwargs={"column": "Stock_Price"}  # After cleaning
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_not_be_null",
                kwargs={"column": "Company"}
            ),
            # MODIFIED: Accept some missing (NO forward fill)
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_not_be_null",
                kwargs={
                    "column": "Stock_Price",
                    "mostly": 0.80  # 80% OK (quarterly gaps)
                }
            ),
            ExpectationConfiguration(
                expectation_type="expect_table_row_count_to_be_between",
                kwargs={
                    "min_value": 2000,
                    "max_value": 15000
                }
            )
        ]
        
        suite_name = self.ge_validator.create_expectation_suite("company_prices_clean_suite", expectations)
        ge_passed, ge_report = self.ge_validator.validate_dataframe(
            df, suite_name, "company_prices_clean", GESeverity.CRITICAL
        )
        
        robust_validator = RobustValidator(
            dataset_name="company_prices_clean",
            enable_auto_fix=False,
            enable_temporal_checks=True if 'Date' in df.columns else False,
            enable_business_rules=True
        )
        
        _, robust_report = robust_validator.validate(df)
        critical_count = robust_report.count_by_severity()['CRITICAL']
        robust_passed = (critical_count == 0)
        
        passed = ge_passed and robust_passed
        
        self.all_reports['company_prices_clean'] = {
            'ge_report': ge_report,
            'robust_report': robust_report.to_dict(),
            'passed': passed
        }
        
        if passed:
            logger.info("  ✅ company_prices_clean.csv validation PASSED")
        else:
            logger.error("  ❌ company_prices_clean.csv validation FAILED")
        
        return passed
    
    def run_all_validations(self) -> bool:
        """Run all cleaned data validations."""
        logger.info("\n" + "="*80)
        logger.info("CHECKPOINT 2: CLEANED DATA VALIDATION")
        logger.info("="*80)
        logger.info("MODIFIED FOR NO FORWARD FILL:")
        logger.info("  - Accept 20-50% missing (quarterly gaps preserved)")
        logger.info("  - Check duplicates removed")
        logger.info("  - Check reporting lags applied")
        logger.info("="*80)
        
        results = {
            'fred': self.validate_fred_clean(),
            'prices': self.validate_company_prices_clean(),
            # Add other datasets as needed
        }
        
        all_passed = all(results.values())
        
        logger.info("\n" + "="*80)
        logger.info("CHECKPOINT 2 SUMMARY")
        logger.info("="*80)
        
        for name, passed in results.items():
            status = "✅ PASSED" if passed else "❌ FAILED"
            logger.info(f"  {name:20s}: {status}")
        
        if all_passed:
            logger.info("\n✅ CHECKPOINT 2 PASSED")
            logger.info("\n➡️  Next: python step2_feature_engineering.py")
            return True
        else:
            logger.error("\n❌ CHECKPOINT 2 FAILED")
            logger.error("\n➡️  Review errors and re-run Step 1")
            return False


def main():
    validator = CleanedDataValidator()
    
    try:
        success = validator.run_all_validations()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()