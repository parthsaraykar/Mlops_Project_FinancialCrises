"""
STEP 4: POST-MERGE DATA CLEANING (NO FORWARD FILL)

Runs AFTER Step 3 (merging)

Purpose:
- Clean merged datasets to address merge-specific issues
- Remove duplicate columns with suffixes (_x, _y, _fred, _market)
- Fix inf values from ratio calculations
- Cap extreme outliers
- Handle missing values WITHOUT forward fill (consistent with Step 1)
- Validate data types
- Remove constant/low-variance columns

CRITICAL: NO FORWARD FILL (consistent with Step 1 cleaning)

Input:
    - data/features/macro_features.csv
    - data/features/merged_features.csv

Output:
    - data/features/macro_features_clean.csv
    - data/features/merged_features_clean.csv

Usage:
    python step4_post_merge_cleaning.py

Next Step:
    Proceed to modeling or validation
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Tuple
import time
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


class PostMergeCleanerNoFFill:
    """
    Post-merge cleaner WITHOUT forward fill.
    
    Consistent with Step 1 philosophy: preserve data sparsity.
    """

    def __init__(self, features_dir: str = "data/features"):
        self.features_dir = Path(features_dir)
        self.reports_dir = Path("data/reports")
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def compute_statistics(self, df: pd.DataFrame, name: str) -> Dict:
        """Compute comprehensive statistics."""
        stats = {
            'dataset_name': name,
            'n_rows': len(df),
            'n_cols': len(df.columns),
            'memory_mb': df.memory_usage(deep=True).sum() / 1024**2,
        }

        if 'Date' in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df['Date']):
                df['Date'] = pd.to_datetime(df['Date'], format='mixed', errors='coerce')
            
            stats['date_min'] = str(df['Date'].min())
            stats['date_max'] = str(df['Date'].max())

        missing = df.isna().sum()
        stats['total_missing'] = missing.sum()
        stats['missing_pct'] = round((missing.sum() / df.size) * 100, 2)

        numeric_df = df.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            inf_count = np.isinf(numeric_df).sum().sum()
            stats['inf_values'] = inf_count

        if 'Date' in df.columns and 'Company' in df.columns:
            stats['duplicates'] = df.duplicated(subset=['Date', 'Company']).sum()
        elif 'Date' in df.columns:
            stats['duplicates'] = df.duplicated(subset=['Date']).sum()
        else:
            stats['duplicates'] = df.duplicated().sum()

        return stats

    def print_statistics_comparison(self, before_stats: Dict, after_stats: Dict):
        """Print before/after comparison."""
        logger.info(f"\n{'='*80}")
        logger.info(f"STATISTICS: {before_stats['dataset_name']}")
        logger.info(f"{'='*80}")

        comparisons = [
            ('Rows', 'n_rows'),
            ('Columns', 'n_cols'),
            ('Total Missing', 'total_missing'),
            ('Missing %', 'missing_pct'),
            ('Inf Values', 'inf_values'),
            ('Duplicates', 'duplicates'),
        ]

        print(f"\n{'Metric':<30} {'BEFORE':>15} {'AFTER':>15} {'Change':>15}")
        print("-" * 75)

        for label, key in comparisons:
            before_val = before_stats.get(key, 'N/A')
            after_val = after_stats.get(key, 'N/A')

            if isinstance(before_val, (int, float)) and isinstance(after_val, (int, float)):
                change = after_val - before_val
                if isinstance(before_val, float):
                    print(f"{label:<30} {before_val:>15.2f} {after_val:>15.2f} {change:>15.2f}")
                else:
                    print(f"{label:<30} {before_val:>15,} {after_val:>15,} {change:>15,}")

    # ========== CLEANING FUNCTIONS ==========

    def remove_duplicate_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate columns from merge."""
        logger.info("\n1. Removing duplicate columns...")
        
        df = df.copy()
        suffixes = ['_x', '_y', '_fred', '_market', '_macro', '_company', '_dup', '_fin']
        
        cols_to_drop = []
        cols_to_rename = {}
        
        for col in df.columns:
            for suffix in suffixes:
                if col.endswith(suffix):
                    base_col = col[:-len(suffix)]
                    if base_col in df.columns:
                        cols_to_drop.append(col)
                    else:
                        cols_to_rename[col] = base_col
        
        if cols_to_drop:
            df.drop(columns=cols_to_drop, inplace=True)
            logger.info(f"   ✓ Removed {len(cols_to_drop)} duplicate columns")
        
        if cols_to_rename:
            df.rename(columns=cols_to_rename, inplace=True)
            logger.info(f"   ✓ Renamed {len(cols_to_rename)} columns")
        
        if not cols_to_drop and not cols_to_rename:
            logger.info(f"   ✓ No duplicate columns found")
        
        return df

    def handle_inf_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Replace inf/-inf with NaN."""
        logger.info("\n2. Handling inf values...")
        
        df = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        inf_before = np.isinf(df[numeric_cols]).sum().sum()
        
        if inf_before > 0:
            logger.info(f"   Found {inf_before:,} inf values")
            df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
            logger.info(f"   ✓ Replaced with NaN")
        else:
            logger.info(f"   ✓ No inf values")
        
        return df

    def cap_extreme_outliers(self, df: pd.DataFrame, 
                            group_col: str = None) -> pd.DataFrame:
        """
        Cap extreme outliers (lenient for quarterly volatility).
        
        Uses 0.1% - 99.9% percentiles (more lenient than 1% - 99%).
        """
        logger.info(f"\n3. Capping extreme outliers (0.1% - 99.9% percentiles)...")
        
        df = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        exclude_cols = ['Date', 'Year', 'Month', 'Day', 'Quarter']
        numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        capped_count = 0
        
        if group_col and group_col in df.columns:
            logger.info(f"   Capping per {group_col}...")
            
            for col in numeric_cols:
                for group_name in df[group_col].unique():
                    group_mask = df[group_col] == group_name
                    group_data = df.loc[group_mask, col]
                    
                    if group_data.notna().sum() > 10:
                        lower = group_data.quantile(0.001)
                        upper = group_data.quantile(0.999)
                        
                        n_capped = ((group_data < lower) | (group_data > upper)).sum()
                        capped_count += n_capped
                        
                        df.loc[group_mask, col] = group_data.clip(lower=lower, upper=upper)
        else:
            for col in numeric_cols:
                if df[col].notna().sum() > 10:
                    lower = df[col].quantile(0.001)
                    upper = df[col].quantile(0.999)
                    
                    n_capped = ((df[col] < lower) | (df[col] > upper)).sum()
                    capped_count += n_capped
                    
                    df[col] = df[col].clip(lower=lower, upper=upper)
        
        if capped_count > 0:
            logger.info(f"   ✓ Capped {capped_count:,} extreme values")
        else:
            logger.info(f"   ✓ No extreme outliers")
        
        return df

    def handle_missing_values_NO_FFILL(self, df: pd.DataFrame, 
                                        group_col: str = None) -> pd.DataFrame:
        """
        MODIFIED: Handle missing WITHOUT forward fill.
        
        Only uses median imputation for moderate missingness (10-50%).
        Consistent with Step 1 philosophy.
        """
        logger.info("\n4. Handling missing values (NO FORWARD FILL)...")
        
        df = df.copy()
        original_missing = df.isna().sum().sum()
        original_missing_pct = (original_missing / df.size) * 100
        
        logger.info(f"   Missing before: {original_missing:,} ({original_missing_pct:.2f}%)")
        
        # Identify columns with moderate missingness (10-50%)
        missing_pct = (df.isna().sum() / len(df)) * 100
        cols_to_fill = missing_pct[(missing_pct > 10) & (missing_pct < 50)].index
        
        logger.info(f"   Columns with 10-50% missing (will fill): {len(cols_to_fill)}")
        
        # Flag columns with >50% missing
        high_missing = missing_pct[missing_pct > 50]
        if len(high_missing) > 0:
            logger.warning(f"   ⚠️  {len(high_missing)} columns with >50% missing (keeping as-is)")
        
        if len(cols_to_fill) > 0:
            if group_col and group_col in df.columns:
                logger.info(f"   Filling with group median per {group_col}...")
                
                for col in cols_to_fill:
                    if pd.api.types.is_numeric_dtype(df[col]):
                        for company in df[group_col].unique():
                            mask = df[group_col] == company
                            median_val = df.loc[mask, col].median()
                            if not pd.isna(median_val):
                                df.loc[mask, col] = df.loc[mask, col].fillna(median_val)
            else:
                logger.info(f"   Filling with global median...")
                
                for col in cols_to_fill:
                    if pd.api.types.is_numeric_dtype(df[col]):
                        median_val = df[col].median()
                        if not pd.isna(median_val):
                            df[col] = df[col].fillna(median_val)
        
        final_missing = df.isna().sum().sum()
        final_missing_pct = (final_missing / df.size) * 100
        filled = original_missing - final_missing
        
        logger.info(f"   ✓ Filled {filled:,} values using median")
        logger.info(f"   Remaining: {final_missing:,} ({final_missing_pct:.2f}%)")
        logger.info(f"   Note: Remaining nulls are EXPECTED (no forward fill)")
        
        return df

    def validate_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and fix data types."""
        logger.info("\n5. Validating data types...")
        
        df = df.copy()
        conversions = []
        
        if 'Date' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['Date']):
            df['Date'] = pd.to_datetime(df['Date'], format='mixed', errors='coerce')
            conversions.append("Date → datetime")
        
        categorical_cols = ['Company', 'Sector', 'Company_Name']
        for col in categorical_cols:
            if col in df.columns and df[col].dtype == 'object':
                df[col] = df[col].astype('category')
                conversions.append(f"{col} → category")
        
        if conversions:
            logger.info(f"   ✓ Converted {len(conversions)} columns")
        else:
            logger.info(f"   ✓ All types correct")
        
        return df

    def remove_constant_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove columns with no variance."""
        logger.info("\n6. Removing constant columns...")
        
        df = df.copy()
        constant_cols = []
        
        for col in df.columns:
            if col not in ['Date', 'Company', 'Sector', 'Company_Name']:
                unique_count = df[col].nunique()
                if unique_count <= 1:
                    constant_cols.append(col)
        
        if constant_cols:
            df.drop(columns=constant_cols, inplace=True)
            logger.info(f"   ✓ Removed {len(constant_cols)} constant columns")
        else:
            logger.info(f"   ✓ No constant columns")
        
        return df

    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate rows."""
        logger.info("\n7. Removing duplicates...")
        
        df = df.copy()
        original_rows = len(df)
        
        if 'Date' in df.columns and 'Company' in df.columns:
            duplicates = df.duplicated(subset=['Date', 'Company']).sum()
            if duplicates > 0:
                df = df.drop_duplicates(subset=['Date', 'Company'], keep='first')
                logger.info(f"   ✓ Removed {duplicates:,} duplicates")
            else:
                logger.info(f"   ✓ No duplicates")
        elif 'Date' in df.columns:
            duplicates = df.duplicated(subset=['Date']).sum()
            if duplicates > 0:
                df = df.drop_duplicates(subset=['Date'], keep='first')
                logger.info(f"   ✓ Removed {duplicates:,} duplicates")
            else:
                logger.info(f"   ✓ No duplicates")
        
        return df

    # ========== MAIN CLEANING ==========

    def clean_macro_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict, Dict]:
        """Clean macro_features.csv (FRED + Market)."""
        logger.info("\n" + "="*80)
        logger.info("CLEANING MACRO_FEATURES.CSV (NO FORWARD FILL)")
        logger.info("="*80)
        
        before_stats = self.compute_statistics(df, 'macro_features')
        
        logger.info(f"\nBEFORE: {df.shape}, Missing: {before_stats['missing_pct']:.2f}%")

        df = self.remove_duplicate_columns(df)
        df = self.handle_inf_values(df)
        df = self.cap_extreme_outliers(df)
        df = self.handle_missing_values_NO_FFILL(df)  # NO FORWARD FILL
        df = self.validate_data_types(df)
        df = self.remove_constant_columns(df)
        df = self.remove_duplicates(df)

        after_stats = self.compute_statistics(df, 'macro_features')
        
        logger.info(f"\nAFTER: {df.shape}, Missing: {after_stats['missing_pct']:.2f}%")
        
        return df, before_stats, after_stats

    def clean_merged_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict, Dict]:
        """Clean merged_features.csv (Macro + Market + Company)."""
        logger.info("\n" + "="*80)
        logger.info("CLEANING MERGED_FEATURES.CSV (NO FORWARD FILL)")
        logger.info("="*80)
        
        before_stats = self.compute_statistics(df, 'merged_features')
        
        logger.info(f"\nBEFORE: {df.shape}, Missing: {before_stats['missing_pct']:.2f}%")

        df = self.remove_duplicate_columns(df)
        df = self.handle_inf_values(df)
        df = self.cap_extreme_outliers(df, group_col='Company')
        df = self.handle_missing_values_NO_FFILL(df, group_col='Company')  # NO FORWARD FILL
        df = self.validate_data_types(df)
        df = self.remove_constant_columns(df)
        df = self.remove_duplicates(df)

        after_stats = self.compute_statistics(df, 'merged_features')
        
        logger.info(f"\nAFTER: {df.shape}, Missing: {after_stats['missing_pct']:.2f}%")
        
        return df, before_stats, after_stats

    # ========== MAIN PIPELINE ==========

    def run_post_merge_cleaning(self):
        """Execute post-merge cleaning (NO FORWARD FILL)."""
        logger.info("\n" + "="*80)
        logger.info("STEP 4: POST-MERGE CLEANING (NO FORWARD FILL)")
        logger.info("="*80)
        logger.info("\nKey Principle: NO FORWARD FILL (consistent with Step 1)")
        logger.info("  ✓ Only median imputation for 10-50% missing")
        logger.info("  ✓ Accept 20-40% final missing (quarterly gaps)")
        logger.info("  ✓ Cap outliers (0.1% - 99.9%)")
        logger.info("="*80)

        overall_start = time.time()
        all_stats = {}

        # Clean macro features
        macro_path = self.features_dir / 'macro_features.csv'
        if macro_path.exists():
            logger.info(f"\n[1/2] LOADING macro_features.csv")
            df_macro = pd.read_csv(macro_path, parse_dates=['Date'])
            
            df_macro_clean, before_macro, after_macro = self.clean_macro_features(df_macro)
            
            output_path = self.features_dir / 'macro_features_clean.csv'
            df_macro_clean.to_csv(output_path, index=False)
            logger.info(f"\n✓ Saved: {output_path}")
            
            all_stats['macro'] = {'before': before_macro, 'after': after_macro}

        # Clean merged features
        merged_path = self.features_dir / 'merged_features.csv'
        if merged_path.exists():
            logger.info(f"\n[2/2] LOADING merged_features.csv")
            df_merged = pd.read_csv(merged_path, parse_dates=['Date'])
            
            df_merged_clean, before_merged, after_merged = self.clean_merged_features(df_merged)
            
            output_path = self.features_dir / 'merged_features_clean.csv'
            df_merged_clean.to_csv(output_path, index=False)
            logger.info(f"\n✓ Saved: {output_path}")
            
            all_stats['merged'] = {'before': before_merged, 'after': after_merged}

        if not all_stats:
            logger.error("\n❌ No datasets cleaned!")
            return None

        # Print comparisons
        logger.info("\n\n" + "="*80)
        logger.info("BEFORE vs AFTER COMPARISON")
        logger.info("="*80)
        
        for name, stats in all_stats.items():
            self.print_statistics_comparison(stats['before'], stats['after'])

        elapsed = time.time() - overall_start

        logger.info("\n" + "="*80)
        logger.info("POST-MERGE CLEANING COMPLETE")
        logger.info("="*80)
        logger.info(f"\n⏱️  Time: {elapsed:.1f}s")
        logger.info(f"\n✅ Step 4 Complete (NO FORWARD FILL)!")
        logger.info(f"\n➡️  Data ready for modeling")

        return all_stats


def main():
    """Execute Step 4: Post-Merge Cleaning."""
    
    cleaner = PostMergeCleanerNoFFill(features_dir="data/features")
    
    try:
        stats = cleaner.run_post_merge_cleaning()
        
        if stats:
            logger.info("\n✅ STEP 4 SUCCESSFULLY COMPLETED")
            return stats
        else:
            logger.error("\n❌ Cleaning failed")
            return None
        
    except FileNotFoundError as e:
        logger.error(f"\n❌ ERROR: {e}")
        logger.error("Run Step 3 first: python step3_data_merging.py")
        return None
    except Exception as e:
        logger.error(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    stats = main()