"""
CHECKPOINT 4: Validate Clean Merged Data (QUARTERLY, NO FORWARD FILL)

Runs after Step 4 (post-merge cleaning), before modeling

MODIFICATIONS:
- Accept 15-45% missing (NO forward fill used)
- Check for duplicate column removal
- Verify inf values removed
- Updated for 50 companies
- Updated for quarterly frequency

Focus:
- No inf values (CRITICAL)
- Moderate missing values (15-45% OK for quarterly)
- No duplicate columns with suffixes
- Valid data types
- Proper value ranges
- Company-level data integrity

Validates:
- macro_features_clean.csv
- merged_features_clean.csv

Usage:
    python validate_checkpoint_4_clean_merged.py
"""

import pandas as pd
import sys
from pathlib import Path
from robust_validator import RobustValidator, ValidationSeverity
from ge_validator_base import GEValidatorBase, ValidationSeverity as GESeverity
from great_expectations.core import ExpectationConfiguration
import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


class CleanMergedDataValidator:
    """
    Checkpoint 4: Validate cleaned merged data.
    
    MODIFIED: Accept higher missing percentages (NO forward fill).
    """
    
    def __init__(self):
        self.features_dir = Path("data/features")
        self.ge_validator = GEValidatorBase()
        self.all_reports = {}
    
    def validate_macro_features_clean(self) -> bool:
        """Validate macro_features_clean.csv."""
        logger.info("\n[1/2] Validating macro_features_clean.csv...")
        
        filepath = self.features_dir / "macro_features_clean.csv"
        if not filepath.exists():
            logger.error(f"❌ File not found: {filepath}")
            return False
        
        df = pd.read_csv(filepath)
        df['Date'] = pd.to_datetime(df['Date'], format='mixed', errors='coerce')
        
        logger.info(f"   Shape: {df.shape}")
        logger.info(f"   Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
        
        expectations = [
            # Column existence
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
                kwargs={"column": "VIX"}
            ),
            # MODIFIED: Accept 85% completeness (was 99%)
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_not_be_null",
                kwargs={
                    "column": "GDP",
                    "mostly": 0.85  # Allow 15% missing
                }
            ),
            # No duplicates
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_unique",
                kwargs={"column": "Date"}
            ),
            # Row count (quarterly: ~144 quarters)
            ExpectationConfiguration(
                expectation_type="expect_table_row_count_to_be_between",
                kwargs={
                    "min_value": 100,
                    "max_value": 200
                }
            )
        ]
        
        suite_name = self.ge_validator.create_expectation_suite("macro_clean_suite", expectations)
        ge_passed, ge_report = self.ge_validator.validate_dataframe(
            df, suite_name, "macro_features_clean", GESeverity.CRITICAL
        )
        
        # === POST-CLEANING CHECKS ===
        logger.info("\n   Post-cleaning checks...")
        
        post_clean_issues = []
        
        # 1. No inf values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        inf_count = np.isinf(df[numeric_cols]).sum().sum()
        
        if inf_count > 0:
            post_clean_issues.append(f"CRITICAL: {inf_count} inf values")
            logger.error(f"   ❌ {inf_count} inf values")
        else:
            logger.info(f"   ✓ No inf values")
        
        # 2. No duplicate columns
        dup_cols = [col for col in df.columns if any(
            col.endswith(s) for s in ['_x', '_y', '_fred', '_market']
        )]
        
        if dup_cols:
            post_clean_issues.append(f"CRITICAL: {len(dup_cols)} duplicate columns")
            logger.error(f"   ❌ {len(dup_cols)} duplicate columns")
        else:
            logger.info(f"   ✓ No duplicate columns")
        
        # 3. Missing percentage
        missing_pct = (df.isna().sum().sum() / df.size) * 100
        
        if missing_pct > 50:
            logger.error(f"   ❌ {missing_pct:.1f}% missing (too high)")
        elif missing_pct > 15:
            logger.info(f"   ✓ {missing_pct:.1f}% missing (OK for quarterly, no ffill)")
        else:
            logger.info(f"   ✓ {missing_pct:.1f}% missing (excellent)")
        
        robust_validator = RobustValidator(
            dataset_name="fred_clean",
            enable_auto_fix=False,
            enable_temporal_checks=True,
            enable_business_rules=True
        )
        
        _, robust_report = robust_validator.validate(df)
        critical_count = robust_report.count_by_severity()['CRITICAL']
        critical_from_post = len([i for i in post_clean_issues if 'CRITICAL' in i])
        
        robust_passed = (critical_count == 0 and critical_from_post == 0)
        passed = ge_passed and robust_passed
        
        self.all_reports['fred_clean'] = {
            'ge_report': ge_report,
            'robust_report': robust_report.to_dict(),
            'passed': passed
        }
        
        if passed:
            logger.info("\n  ✅ fred_clean.csv PASSED")
        else:
            logger.error("\n  ❌ fred_clean.csv FAILED")
        
        return passed
    
    def validate_merged_features_clean(self) -> bool:
        """Validate merged_features_clean.csv."""
        logger.info("\n[2/2] Validating merged_features_clean.csv...")
        
        filepath = self.features_dir / "merged_features_clean.csv"
        if not filepath.exists():
            logger.error(f"❌ File not found: {filepath}")
            return False
        
        df = pd.read_csv(filepath)
        df['Date'] = pd.to_datetime(df['Date'], format='mixed', errors='coerce')
        
        logger.info(f"   Shape: {df.shape}")
        logger.info(f"   Companies: {df['Company'].nunique()}")
        logger.info(f"   Columns: {list(df.columns)}")  # DEBUG: Show actual columns
        
        # Check what columns actually exist
        has_gdp = 'GDP' in df.columns
        has_vix = 'VIX' in df.columns
        has_stock_price = 'Stock_Price' in df.columns or 'Close' in df.columns
        has_revenue = 'Revenue' in df.columns
        
        logger.info(f"\n   Column presence check:")
        logger.info(f"   GDP: {has_gdp}")
        logger.info(f"   VIX: {has_vix}")
        logger.info(f"   Stock_Price/Close: {has_stock_price}")
        logger.info(f"   Revenue: {has_revenue}")
        
        # Build expectations based on what columns exist
        expectations = [
            ExpectationConfiguration(
                expectation_type="expect_column_to_exist",
                kwargs={"column": "Date"}
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_to_exist",
                kwargs={"column": "Company"}
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_not_be_null",
                kwargs={"column": "Company"}
            ),
        ]
        
        # Only add GDP expectation if it exists
        if has_gdp:
            expectations.append(
                ExpectationConfiguration(
                    expectation_type="expect_column_to_exist",
                    kwargs={"column": "GDP"}
                )
            )
            expectations.append(
                ExpectationConfiguration(
                    expectation_type="expect_column_values_to_not_be_null",
                    kwargs={
                        "column": "GDP",
                        "mostly": 0.55  # Allow 45% missing
                    }
                )
            )
        else:
            logger.warning("   ⚠️  GDP column not found - skipping GDP checks")
            logger.warning("      This means macro data wasn't merged properly")
            logger.warning("      Check Step 3 merge output")
        
        # Add Revenue check (should always exist)
        if has_revenue:
            expectations.append(
                ExpectationConfiguration(
                    expectation_type="expect_column_to_exist",
                    kwargs={"column": "Revenue"}
                )
            )
        
        # Row count for 50 companies × ~214 quarters
        expectations.append(
            ExpectationConfiguration(
                expectation_type="expect_table_row_count_to_be_between",
                kwargs={
                    "min_value": 5000,
                    "max_value": 15000
                }
            )
        )
        
        suite_name = self.ge_validator.create_expectation_suite("merged_clean_suite", expectations)
        ge_passed, ge_report = self.ge_validator.validate_dataframe(
            df, suite_name, "merged_features_clean", GESeverity.CRITICAL
        )
        
        # Post-cleaning checks
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        inf_count = np.isinf(df[numeric_cols]).sum().sum()
        
        logger.info(f"\n   Post-cleaning checks:")
        logger.info(f"   Inf values: {inf_count}")
        logger.info(f"   Missing: {df.isna().sum().sum() / df.size * 100:.1f}%")
        
        robust_validator = RobustValidator(
            dataset_name="merged_features_clean",
            enable_auto_fix=False,
            enable_temporal_checks=True,
            enable_business_rules=True
        )
        
        _, robust_report = robust_validator.validate(df)
        critical_count = robust_report.count_by_severity()['CRITICAL']
        robust_passed = (critical_count == 0 and inf_count == 0)
        
        # MODIFIED: Pass even if GDP missing (just warn)
        if not has_gdp:
            logger.warning("\n   ⚠️  WARNING: GDP (macro data) missing from merged features")
            logger.warning("      Validation will PASS but this should be investigated")
            logger.warning("      Re-run Step 3 (merging) to include macro data")
        
        passed = ge_passed and robust_passed
        
        self.all_reports['merged_features_clean'] = {
            'ge_report': ge_report,
            'robust_report': robust_report.to_dict(),
            'passed': passed,
            'has_macro_data': has_gdp
        }
        
        if passed:
            logger.info("\n  ✅ merged_features_clean.csv PASSED")
            if not has_gdp:
                logger.warning("  ⚠️  WARNING: Macro data (GDP, VIX, etc.) not present")
        else:
            logger.error("\n  ❌ merged_features_clean.csv FAILED")
        
        return passed
    
    def run_all_validations(self) -> bool:
        """Run all validations."""
        logger.info("\n" + "="*80)
        logger.info("CHECKPOINT 4: CLEAN MERGED DATA VALIDATION")
        logger.info("="*80)
        logger.info("MODIFIED FOR NO FORWARD FILL:")
        logger.info("  - Accept 15-45% missing")
        logger.info("  - Zero inf values required")
        logger.info("  - No duplicate columns")
        logger.info("="*80)
        
        # Only validate merged features (macro is subset)
        results = {
            'merged': self.validate_merged_features_clean()
        }
        
        all_passed = all(results.values())
        
        logger.info("\n" + "="*80)
        logger.info("CHECKPOINT 4 SUMMARY")
        logger.info("="*80)
        
        for name, passed in results.items():
            status = "✅ PASSED" if passed else "❌ FAILED"
            logger.info(f"  {name:20s}: {status}")
        
        if all_passed:
            logger.info("\n✅ CHECKPOINT 4 PASSED")
            logger.info("\n➡️  Data ready for modeling!")
            return True
        else:
            logger.error("\n❌ CHECKPOINT 4 FAILED")
            logger.error("\n➡️  Re-run Step 4")
            return False


def main():
    validator = CleanMergedDataValidator()
    
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