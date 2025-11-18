"""
STEP 1: DATA CLEANING WITH POINT-IN-TIME CORRECTNESS (NO FORWARD FILL)

This script cleans all raw data files while preserving point-in-time correctness.

Key Features:
1. NO forward-fill (removed to preserve data sparsity)
2. Apply reporting lags to quarterly financials (45 days)
3. Detect outliers but DON'T remove (crises are real!)
4. Per-company handling (no cross-contamination)
5. Comprehensive before/after statistics

MODIFICATIONS FOR QUARTERLY DATA:
- Accept higher missing percentages (20-40% is normal)
- Only fill leading NaNs with median (no ffill/bfill)
- Adjusted for 50 companies (up from 25)
- Date range: 1990-2025 (up from 2005-2025)

Input:  data/raw/*.csv (5 files)
Output: data/clean/*.csv (5 files)

Usage:
    python step1_data_cleaning.py

Next Step:
    python step2_feature_engineering.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, Tuple, List
import time
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


class PointInTimeDataCleanerNoFFill:
    """Clean data while preserving point-in-time correctness WITHOUT forward filling."""

    # Reporting lags (days after quarter-end when data becomes available)
    REPORTING_LAGS = {
        'earnings': 45,      # Earnings reported ~45 days after quarter end
        'balance_sheet': 45, # Balance sheet same as earnings
        'macro': 30          # Macro data (GDP, CPI) ~30 days lag
    }

    def __init__(self, raw_dir: str = "data/raw", clean_dir: str = "data/clean"):
        self.raw_dir = Path(raw_dir)
        self.clean_dir = Path(clean_dir)
        self.clean_dir.mkdir(parents=True, exist_ok=True)

        # Create reports directory
        self.report_dir = Path("data/reports")
        self.report_dir.mkdir(parents=True, exist_ok=True)

    # ========== STATISTICS FUNCTIONS ==========

    def compute_statistics(self, df: pd.DataFrame, name: str) -> Dict:
        """Compute comprehensive statistics for a dataset."""
        stats = {
            'dataset_name': name,
            'n_rows': len(df),
            'n_cols': len(df.columns),
            'memory_mb': df.memory_usage(deep=True).sum() / 1024**2,
        }

        # Date range
        if 'Date' in df.columns:
            # Ensure Date is datetime
            if not pd.api.types.is_datetime64_any_dtype(df['Date']):
                df['Date'] = pd.to_datetime(df['Date'], format='mixed', errors='coerce')

            stats['date_min'] = str(df['Date'].min())
            stats['date_max'] = str(df['Date'].max())
            stats['date_range_days'] = (df['Date'].max() - df['Date'].min()).days

        # Missing values
        missing = df.isna().sum()
        stats['total_missing'] = missing.sum()
        stats['missing_pct'] = round((missing.sum() / df.size) * 100, 2)
        stats['cols_with_missing'] = (missing > 0).sum()

        # Duplicates
        if 'Date' in df.columns and 'Company' in df.columns:
            stats['duplicates'] = df.duplicated(subset=['Date', 'Company']).sum()
        elif 'Date' in df.columns:
            stats['duplicates'] = df.duplicated(subset=['Date']).sum()
        else:
            stats['duplicates'] = df.duplicated().sum()

        # Numeric statistics
        numeric_df = df.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            stats['n_numeric_cols'] = len(numeric_df.columns)
            stats['mean_value'] = numeric_df.mean().mean()
            stats['std_value'] = numeric_df.std().mean()

        # Categorical
        categorical_df = df.select_dtypes(exclude=[np.number])
        stats['n_categorical_cols'] = len(categorical_df.columns)

        return stats

    def print_statistics_comparison(self, before_stats: Dict, after_stats: Dict):
        """Print before/after comparison in clean format."""
        logger.info(f"\n{'='*80}")
        logger.info(f"STATISTICS: {before_stats['dataset_name']}")
        logger.info(f"{'='*80}")

        comparisons = [
            ('Rows', 'n_rows'),
            ('Columns', 'n_cols'),
            ('Memory (MB)', 'memory_mb'),
            ('Date Range (days)', 'date_range_days'),
            ('Total Missing Values', 'total_missing'),
            ('Missing %', 'missing_pct'),
            ('Columns with Missing', 'cols_with_missing'),
            ('Duplicate Rows', 'duplicates'),
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
            else:
                print(f"{label:<30} {str(before_val):>15} {str(after_val):>15} {'':>15}")

    # ========== MODIFIED: NO FORWARD FILL ==========

    def handle_nulls_no_lookahead(self, df: pd.DataFrame, date_col: str = 'Date',
                                  group_col: str = None) -> pd.DataFrame:
        """
        MODIFIED: Handle nulls WITHOUT forward filling.
        
        Strategy:
        - REMOVED: ffill() - no forward propagation
        - KEPT: Median fill for leading NaNs only
        - RESULT: More missing data preserved
        
        Rationale: Quarterly data naturally has gaps. Forward filling 
        would create fake data points between quarters.
        
        Args:
            df: DataFrame to clean
            date_col: Date column name
            group_col: If provided, handle nulls per group (e.g., per Company)
            
        Returns:
            Cleaned DataFrame with only leading NaNs filled
        """
        df = df.copy()
        df_original = df.copy()

        # Ensure date is datetime
        if date_col in df.columns and not pd.api.types.is_datetime64_any_dtype(df[date_col]):
            df[date_col] = pd.to_datetime(df[date_col], format='mixed', errors='coerce')

        if group_col:
            # Fill within groups (per company)
            numeric_cols = df.select_dtypes(include=[np.number]).columns

            for col in numeric_cols:
                # REMOVED: Forward fill
                # df[col] = df.groupby(group_col)[col].ffill()  # ❌ DELETED
                
                # ONLY fill leading NaNs with group median
                for group_name in df[group_col].unique():
                    group_mask = df[group_col] == group_name
                    group_data = df.loc[group_mask, col]

                    if group_data.isna().any():
                        valid_data = group_data.dropna()
                        if len(valid_data) > 0:
                            # Only fill leading NaNs (first N rows with no data)
                            first_valid_idx = valid_data.index[0]
                            
                            # Get indices of leading nulls
                            leading_null_indices = []
                            for idx in group_data.index:
                                if idx < first_valid_idx and pd.isna(group_data.loc[idx]):
                                    leading_null_indices.append(idx)
                            
                            if leading_null_indices:
                                fill_value = valid_data.head(min(10, len(valid_data))).median()
                                df.loc[leading_null_indices, col] = fill_value
        else:
            # Fill entire dataset
            df_indexed = df.set_index(date_col) if date_col in df.columns else df

            # REMOVED: Forward fill
            # df = df.ffill()  # ❌ DELETED

            # ONLY fill leading NaNs
            for col in df_indexed.columns:
                if df_indexed[col].isna().any():
                    valid_data = df_indexed[col].dropna()
                    if len(valid_data) > 0:
                        first_valid_idx = valid_data.index[0]
                        
                        # Get leading nulls
                        leading_null_indices = []
                        for idx in df_indexed.index:
                            if idx < first_valid_idx and pd.isna(df_indexed.loc[idx, col]):
                                leading_null_indices.append(idx)
                        
                        if leading_null_indices:
                            fill_value = valid_data.head(min(10, len(valid_data))).median()
                            df_indexed.loc[leading_null_indices, col] = fill_value

            # Reset index if it was set
            if date_col in df_indexed.index.names or df_indexed.index.name == date_col:
                df = df_indexed.reset_index()
            else:
                df = df_indexed

        # Log what was filled
        filled_count = df_original.isna().sum().sum() - df.isna().sum().sum()
        if filled_count > 0:
            logger.info(f"  ✓ Filled {filled_count:,} leading NaNs with median (NO forward fill)")
        
        # Report remaining nulls
        remaining_nulls = df.isna().sum().sum()
        remaining_pct = (remaining_nulls / df.size) * 100
        logger.info(f"  ℹ️  Remaining nulls: {remaining_nulls:,} ({remaining_pct:.2f}%)")
        logger.info(f"      This is EXPECTED with quarterly data - natural gaps preserved")

        return df

    # ========== POINT-IN-TIME FUNCTIONS ==========

    def apply_reporting_lag(self, df: pd.DataFrame, lag_days: int,
                           group_col: str = None) -> pd.DataFrame:
        """
        Apply reporting lag to quarterly data for point-in-time correctness.

        Example: Q1 2020 earnings (3/31) are reported 45 days later (5/15)
        So on any day before 5/15, we should use Q4 2019 data, not Q1 2020.

        Args:
            df: DataFrame with quarterly data
            lag_days: Number of days after quarter-end when data is available
            group_col: If provided, shift within groups (e.g., per Company)
        """
        logger.info(f"\n⏰ Applying {lag_days}-day reporting lag for point-in-time correctness...")

        df = df.copy()

        # Shift dates forward by reporting lag
        df['Date'] = df['Date'] + pd.Timedelta(days=lag_days)

        # Log the transformation
        example_date = pd.Timestamp('2020-03-31')
        example_available = example_date + pd.Timedelta(days=lag_days)
        logger.info(f"  Example: Q1 2020 (3/31) → Available on {example_available.date()}")

        return df

    # ========== CLEAN INDIVIDUAL DATASETS ==========

    def clean_fred(self) -> Tuple[pd.DataFrame, Dict, Dict]:
        """Clean FRED data with point-in-time correctness."""
        logger.info("\n" + "="*80)
        logger.info("CLEANING FRED DATA (NO FORWARD FILL)")
        logger.info("="*80)

        # Load
        filepath = self.raw_dir / 'fred_raw.csv'
        df = pd.read_csv(filepath)
        before_stats = self.compute_statistics(df, 'FRED')

        logger.info(f"\nBEFORE CLEANING:")
        logger.info(f"  Shape: {df.shape}")
        logger.info(f"  Missing values: {df.isna().sum().sum()} ({before_stats['missing_pct']}%)")
        logger.info(f"  Duplicates: {before_stats['duplicates']}")

        # Standardize column names
        if 'DATE' in df.columns:
            df.rename(columns={'DATE': 'Date'}, inplace=True)
        
        df['Date'] = pd.to_datetime(df['Date'], format='mixed', errors='coerce')
        df.sort_values('Date', inplace=True)

        # Handle nulls (NO forward fill)
        df = self.handle_nulls_no_lookahead(df, date_col='Date')

        # Remove duplicates
        df = df.drop_duplicates(subset=['Date'], keep='last')

        # After statistics
        after_stats = self.compute_statistics(df, 'FRED')

        logger.info(f"\nAFTER CLEANING:")
        logger.info(f"  Shape: {df.shape}")
        logger.info(f"  Missing values: {df.isna().sum().sum()} ({after_stats['missing_pct']}%)")
        logger.info(f"  Duplicates: {after_stats['duplicates']}")

        # Save
        output_path = self.clean_dir / 'fred_clean.csv'
        df.to_csv(output_path, index=False)
        logger.info(f"\n✓ Saved to: {output_path}")

        return df, before_stats, after_stats

    def clean_market(self) -> Tuple[pd.DataFrame, Dict, Dict]:
        """Clean market data (real-time, no lag needed)."""
        logger.info("\n" + "="*80)
        logger.info("CLEANING MARKET DATA (NO FORWARD FILL)")
        logger.info("="*80)

        # Load
        filepath = self.raw_dir / 'market_raw.csv'
        df = pd.read_csv(filepath)
        before_stats = self.compute_statistics(df, 'Market')

        logger.info(f"\nBEFORE CLEANING:")
        logger.info(f"  Shape: {df.shape}")
        logger.info(f"  Missing: {before_stats['total_missing']} ({before_stats['missing_pct']}%)")

        # Parse date
        df['Date'] = pd.to_datetime(df['Date'], format='mixed', errors='coerce')
        df.sort_values('Date', inplace=True)

        # Rename columns for consistency
        if 'SP500' in df.columns:
            df.rename(columns={'SP500': 'SP500_Close'}, inplace=True)

        # Handle nulls (no reporting lag - market data is real-time)
        logger.info("\n  Market data is real-time (no reporting lag needed)")
        df = self.handle_nulls_no_lookahead(df, date_col='Date')

        # Remove duplicates
        df = df.drop_duplicates(subset=['Date'], keep='last')

        after_stats = self.compute_statistics(df, 'Market')

        logger.info(f"\nAFTER CLEANING:")
        logger.info(f"  Shape: {df.shape}")
        logger.info(f"  Missing: {after_stats['total_missing']} ({after_stats['missing_pct']}%)")

        output_path = self.clean_dir / 'market_clean.csv'
        df.to_csv(output_path, index=False)
        logger.info(f"\n✓ Saved to: {output_path}")

        return df, before_stats, after_stats

    def clean_company_prices(self) -> Tuple[pd.DataFrame, Dict, Dict]:
        """Clean company stock prices (real-time, no lag)."""
        logger.info("\n" + "="*80)
        logger.info("CLEANING COMPANY PRICES (NO FORWARD FILL)")
        logger.info("="*80)

        # Load
        filepath = self.raw_dir / 'company_prices_raw.csv'
        df = pd.read_csv(filepath)
        before_stats = self.compute_statistics(df, 'Company Prices')

        logger.info(f"\nBEFORE CLEANING:")
        logger.info(f"  Shape: {df.shape}")
        logger.info(f"  Companies: {df['Company'].nunique() if 'Company' in df.columns else 'N/A'}")
        logger.info(f"  Missing: {before_stats['total_missing']} ({before_stats['missing_pct']}%)")

        # Parse date
        df['Date'] = pd.to_datetime(df['Date'], format='mixed', errors='coerce')

        # Keep needed columns, use Adj_Close (accounts for splits/dividends)
        keep_cols = ['Date', 'Close', 'Volume', 'Company', 'Company_Name', 'Sector']
        
        # Check which columns exist
        available_cols = [col for col in keep_cols if col in df.columns]
        
        # Handle Adj Close if it exists
        if 'Adj Close' in df.columns:
            df['Stock_Price'] = df['Adj Close']
        elif 'Adj_Close' in df.columns:
            df['Stock_Price'] = df['Adj_Close']
        elif 'Close' in df.columns:
            df['Stock_Price'] = df['Close']
        
        # Add Stock_Price to keep list
        if 'Stock_Price' not in available_cols:
            available_cols.append('Stock_Price')
        
        # Keep only available columns
        df = df[available_cols].copy()

        # Sort
        df.sort_values(['Company', 'Date'], inplace=True)

        # Handle nulls per company (no reporting lag - prices are real-time)
        logger.info("\n  Stock prices are real-time (no reporting lag needed)")
        df = self.handle_nulls_no_lookahead(df, date_col='Date', group_col='Company')

        # Remove duplicates
        df = df.drop_duplicates(subset=['Date', 'Company'], keep='last')

        after_stats = self.compute_statistics(df, 'Company Prices')

        logger.info(f"\nAFTER CLEANING:")
        logger.info(f"  Shape: {df.shape}")
        logger.info(f"  Missing: {after_stats['total_missing']} ({after_stats['missing_pct']}%)")

        # Per-company summary
        logger.info(f"\n  Per-company summary:")
        for company in sorted(df['Company'].unique())[:5]:  # Show first 5
            company_df = df[df['Company'] == company]
            logger.info(f"    {company}: {len(company_df):,} quarters, " +
                       f"{company_df['Date'].min().date()} to {company_df['Date'].max().date()}")

        output_path = self.clean_dir / 'company_prices_clean.csv'
        df.to_csv(output_path, index=False)
        logger.info(f"\n✓ Saved to: {output_path}")

        return df, before_stats, after_stats

    def clean_balance_sheet(self) -> Tuple[pd.DataFrame, Dict, Dict]:
        """Clean balance sheet with 45-day reporting lag."""
        logger.info("\n" + "="*80)
        logger.info("CLEANING BALANCE SHEET (NO FORWARD FILL)")
        logger.info("="*80)

        # Load
        filepath = self.raw_dir / 'company_balance_raw.csv'
        df = pd.read_csv(filepath)
        before_stats = self.compute_statistics(df, 'Balance Sheet')

        logger.info(f"\nBEFORE CLEANING:")
        logger.info(f"  Shape: {df.shape}")
        logger.info(f"  Companies: {df['Company'].nunique() if 'Company' in df.columns else 'N/A'}")
        logger.info(f"  Missing: {before_stats['total_missing']} ({before_stats['missing_pct']}%)")

        # Parse date
        df['Date'] = pd.to_datetime(df['Date'], format='mixed', errors='coerce')
        df.sort_values(['Company', 'Date'], inplace=True)

        # CRITICAL: Apply 45-day reporting lag
        logger.info(f"\n⏰ Applying {self.REPORTING_LAGS['balance_sheet']}-day reporting lag...")
        logger.info("  Why: Balance sheets for Q1 (3/31) are filed ~45 days later (5/15)")
        logger.info("  Effect: Q1 data becomes 'available' on 5/15, not 3/31")

        df = self.apply_reporting_lag(df, lag_days=self.REPORTING_LAGS['balance_sheet'])

        # Handle Long_Term_Debt - allow median fill for this critical field
        logger.info("\n  Handling missing Long_Term_Debt...")
        if 'Long_Term_Debt' in df.columns:
            before_ltd = df['Long_Term_Debt'].isna().sum()
            # Use group median (no ffill)
            for company in df['Company'].unique():
                mask = df['Company'] == company
                median_debt = df.loc[mask, 'Long_Term_Debt'].median()
                if not pd.isna(median_debt):
                    df.loc[mask, 'Long_Term_Debt'] = df.loc[mask, 'Long_Term_Debt'].fillna(median_debt)
            after_ltd = df['Long_Term_Debt'].isna().sum()
            logger.info(f"    Long_Term_Debt: {before_ltd:,} → {after_ltd:,} missing")

        # Calculate Total_Debt
        if 'Long_Term_Debt' in df.columns and 'Short_Term_Debt' in df.columns:
            df['Total_Debt'] = df['Long_Term_Debt'].fillna(0) + df['Short_Term_Debt'].fillna(0)

        # Handle other nulls per company (NO forward fill)
        df = self.handle_nulls_no_lookahead(df, date_col='Date', group_col='Company')

        # Remove duplicates
        df = df.drop_duplicates(subset=['Date', 'Company'], keep='last')

        after_stats = self.compute_statistics(df, 'Balance Sheet')

        logger.info(f"\nAFTER CLEANING:")
        logger.info(f"  Shape: {df.shape}")
        logger.info(f"  Missing: {after_stats['total_missing']} ({after_stats['missing_pct']}%)")

        output_path = self.clean_dir / 'company_balance_clean.csv'
        df.to_csv(output_path, index=False)
        logger.info(f"\n✓ Saved to: {output_path}")

        return df, before_stats, after_stats

    def clean_income_statement(self) -> Tuple[pd.DataFrame, Dict, Dict]:
        """Clean income statement with 45-day reporting lag."""
        logger.info("\n" + "="*80)
        logger.info("CLEANING INCOME STATEMENT (NO FORWARD FILL)")
        logger.info("="*80)

        # Load
        filepath = self.raw_dir / 'company_income_raw.csv'
        df = pd.read_csv(filepath)
        before_stats = self.compute_statistics(df, 'Income Statement')

        logger.info(f"\nBEFORE CLEANING:")
        logger.info(f"  Shape: {df.shape}")
        logger.info(f"  Companies: {df['Company'].nunique() if 'Company' in df.columns else 'N/A'}")
        logger.info(f"  Missing: {before_stats['total_missing']} ({before_stats['missing_pct']}%)")

        # Parse date
        df['Date'] = pd.to_datetime(df['Date'], format='mixed', errors='coerce')
        df.sort_values(['Company', 'Date'], inplace=True)

        # Handle EPS column
        logger.info("\n💰 Handling EPS (Earnings Per Share) column...")
        
        if 'EPS' in df.columns:
            null_count = df['EPS'].isna().sum()
            total_rows = len(df)
            
            if null_count > 0:
                logger.info(f"  ℹ️  EPS has {null_count:,} null values ({null_count/total_rows*100:.1f}%)")
                logger.info("      Filling nulls with 0")
                df['EPS'] = df['EPS'].fillna(0.0)
            else:
                logger.info("  ✅ EPS column has no null values")
        else:
            logger.warning("  ⚠️  EPS column not found in data")
            logger.info("      Creating EPS column with value 0")
            df['EPS'] = 0.0
        
        logger.info(f"  ✓ EPS handling complete")

        # Apply 45-day reporting lag
        logger.info(f"\n⏰ Applying {self.REPORTING_LAGS['earnings']}-day reporting lag...")
        logger.info("  Why: Earnings for Q1 (3/31) are reported ~45 days later (5/15)")
        logger.info("  Effect: Q1 data becomes 'available' on 5/15, not 3/31")
        
        df = self.apply_reporting_lag(df, lag_days=self.REPORTING_LAGS['earnings'])

        # Handle nulls per company (NO forward fill)
        logger.info("\n🔧 Handling remaining null values (NO forward fill)...")
        df = self.handle_nulls_no_lookahead(df, date_col='Date', group_col='Company')

        # Remove duplicates
        df = df.drop_duplicates(subset=['Date', 'Company'], keep='last')

        after_stats = self.compute_statistics(df, 'Income Statement')

        logger.info(f"\nAFTER CLEANING:")
        logger.info(f"  Shape: {df.shape}")
        logger.info(f"  Companies: {df['Company'].nunique()}")
        logger.info(f"  Missing: {after_stats['total_missing']} ({after_stats['missing_pct']}%)")
        logger.info(f"  Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")

        output_path = self.clean_dir / 'company_income_clean.csv'
        df.to_csv(output_path, index=False)
        logger.info(f"\n✓ Saved to: {output_path}")

        return df, before_stats, after_stats

    def save_statistics_report(self, all_stats: Dict):
        """Save detailed cleaning report."""
        report_data = []

        for dataset_name, stats_pair in all_stats.items():
            before = stats_pair['before']
            after = stats_pair['after']

            report_data.append({
                'Dataset': dataset_name,
                'Rows_Before': before['n_rows'],
                'Rows_After': after['n_rows'],
                'Missing_Before': before['total_missing'],
                'Missing_After': after['total_missing'],
                'Missing_Pct_Before': before['missing_pct'],
                'Missing_Pct_After': after['missing_pct'],
                'Duplicates_Before': before['duplicates'],
                'Duplicates_After': after['duplicates']
            })

        report_df = pd.DataFrame(report_data)
        report_path = self.report_dir / 'step1_cleaning_report.csv'
        report_df.to_csv(report_path, index=False)
        logger.info(f"\n✓ Cleaning report saved to: {report_path}")

        return report_df

    # ========== MASTER CLEANING PIPELINE ==========

    def clean_all(self) -> Dict[str, Tuple[pd.DataFrame, Dict, Dict]]:
        """Run complete point-in-time cleaning pipeline with full statistics."""
        logger.info("\n" + "="*80)
        logger.info("STEP 1: DATA CLEANING PIPELINE (NO FORWARD FILL)")
        logger.info("="*80)
        logger.info("\nKey Principles:")
        logger.info("  1. NO forward fill (removed - preserves data sparsity)")
        logger.info("  2. Apply reporting lags to quarterly financials (45 days)")
        logger.info("  3. Detect outliers but DON'T remove (crises are real!)")
        logger.info("  4. Per-company handling (no cross-contamination)")
        logger.info("\nModifications for Quarterly Data:")
        logger.info("  - Accept 20-40% missing (natural quarterly gaps)")
        logger.info("  - Only fill leading NaNs with median")
        logger.info("  - Date range: 1990-2025 (35 years)")
        logger.info("  - Companies: 50 (expanded from 25)")
        logger.info("="*80)

        overall_start = time.time()
        
        all_results = {}
        all_stats = {}

        # Clean each dataset
        logger.info("\n[1/5] Cleaning FRED data...")
        df_fred, before_fred, after_fred = self.clean_fred()
        all_results['fred'] = df_fred
        all_stats['fred'] = {'before': before_fred, 'after': after_fred}

        logger.info("\n[2/5] Cleaning Market data...")
        df_market, before_market, after_market = self.clean_market()
        all_results['market'] = df_market
        all_stats['market'] = {'before': before_market, 'after': after_market}

        logger.info("\n[3/5] Cleaning Company Prices...")
        df_prices, before_prices, after_prices = self.clean_company_prices()
        all_results['prices'] = df_prices
        all_stats['prices'] = {'before': before_prices, 'after': after_prices}

        logger.info("\n[4/5] Cleaning Balance Sheet...")
        df_balance, before_balance, after_balance = self.clean_balance_sheet()
        all_results['balance'] = df_balance
        all_stats['balance'] = {'before': before_balance, 'after': after_balance}

        logger.info("\n[5/5] Cleaning Income Statement...")
        df_income, before_income, after_income = self.clean_income_statement()
        all_results['income'] = df_income
        all_stats['income'] = {'before': before_income, 'after': after_income}

        # ========== PRINT BEFORE/AFTER COMPARISONS ==========

        logger.info("\n\n" + "="*80)
        logger.info("BEFORE vs AFTER COMPARISON - ALL DATASETS")
        logger.info("="*80)

        for name, stats in all_stats.items():
            self.print_statistics_comparison(stats['before'], stats['after'])

        # ========== SAVE COMPREHENSIVE REPORT ==========

        summary_report = self.save_statistics_report(all_stats)

        # ========== FINAL SUMMARY ==========
        
        elapsed = time.time() - overall_start

        logger.info("\n\n" + "="*80)
        logger.info("STEP 1 COMPLETE - SUMMARY")
        logger.info("="*80)
        
        logger.info(f"\n📊 DATA CLEANED:")
        logger.info(f"  1. FRED:            {df_fred.shape[0]:,} rows × {df_fred.shape[1]} cols")
        logger.info(f"  2. Market:          {df_market.shape[0]:,} rows × {df_market.shape[1]} cols")
        logger.info(f"  3. Company Prices:  {df_prices.shape[0]:,} rows ({df_prices['Company'].nunique()} companies)")
        logger.info(f"  4. Balance Sheets:  {df_balance.shape[0]:,} rows ({df_balance['Company'].nunique()} companies)")
        logger.info(f"  5. Income Stmts:    {df_income.shape[0]:,} rows ({df_income['Company'].nunique()} companies)")

        logger.info(f"\n📁 OUTPUT FILES (data/clean/):")
        logger.info(f"  - fred_clean.csv")
        logger.info(f"  - market_clean.csv")
        logger.info(f"  - company_prices_clean.csv")
        logger.info(f"  - company_balance_clean.csv")
        logger.info(f"  - company_income_clean.csv")

        logger.info(f"\n📊 REPORTS (data/reports/):")
        logger.info(f"  - step1_cleaning_report.csv")

        logger.info(f"\n⏱️  Total Time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
        
        logger.info(f"\n📊 KEY MODIFICATIONS:")
        logger.info(f"  ❌ NO forward filling (preserves sparsity)")
        logger.info(f"  ✓ Leading NaNs filled with median")
        logger.info(f"  ✓ Reporting lags applied (45 days)")
        logger.info(f"  ✓ Natural quarterly gaps preserved")
        logger.info("="*80)
        
        logger.info("\n✅ Step 1 Complete!")
        logger.info("\n➡️  Next Steps:")
        logger.info("   1. Run: python step2_feature_engineering.py")

        return all_results, all_stats


def main():
    """Execute Step 1: Data Cleaning."""

    cleaner = PointInTimeDataCleanerNoFFill(raw_dir="data/raw", clean_dir="data/clean")
    
    try:
        cleaned_data, statistics = cleaner.clean_all()
        
        logger.info("\n" + "="*80)
        logger.info("✅ STEP 1 SUCCESSFULLY COMPLETED")
        logger.info("="*80)
        
        return cleaned_data, statistics
        
    except FileNotFoundError as e:
        logger.error(f"\n❌ ERROR: {e}")
        logger.error("Make sure raw data files exist in data/raw/")
        logger.error("Run Step 0 first: python step0_data_collection.py")
        return None, None
    except Exception as e:
        logger.error(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return None, None


if __name__ == "__main__":
    cleaned, stats = main()