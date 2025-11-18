"""
STEP 3: DATA MERGING (QUARTERLY-AWARE)

Combine feature datasets into final merged datasets for modeling.

Pipeline 1 (VAE - Scenario Generation):
    macro_features.csv = FRED + Market (resampled to quarterly)
    - Quarterly macro/market data only
    - Used to train VAE for generating stress scenarios

Pipeline 2 (XGBoost/LSTM - Prediction):
    merged_features.csv = FRED + Market + Company
    - Quarterly company-date observations with full macro context
    - Used to train predictive models

Merge Strategy:
- Resample daily market data to quarterly BEFORE merging
- Pipeline 1: Merge FRED + Market on Date (both quarterly)
- Pipeline 2: Merge macro+market with company data on Date
- Handle missing values appropriately (accept 20-40% for quarterly)

Input:  data/features/fred_features.csv, market_features.csv, company_features.csv
Output: data/features/macro_features.csv, merged_features.csv

Usage:
    python step3_data_merging.py

Next Step:
    python step4_post_merge_cleaning.py (if needed)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, Tuple
import time
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


class QuarterlyDataMerger:
    """
    Merge feature-engineered datasets.
    
    SIMPLIFIED: Step 2 already created daily + quarterly features.
    Step 3 just merges company + macro quarterly data.
    """

    def __init__(self, features_dir: str = "data/features"):
        self.features_dir = Path(features_dir)
        self.reports_dir = Path("data/reports")
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def load_feature_datasets(self) -> Dict[str, pd.DataFrame]:
        """Load all feature datasets from Step 2."""
        logger.info("="*80)
        logger.info("LOADING FEATURE DATASETS FROM STEP 2")
        logger.info("="*80)

        data = {}

        # Check what files Step 2 created
        macro_daily_path = self.features_dir / 'macro_features_daily.csv'
        macro_quarterly_path = self.features_dir / 'macro_features_quarterly.csv'
        company_path = self.features_dir / 'company_features.csv'

        # Load macro_features_daily (for VAE)
        if macro_daily_path.exists():
            data['macro_daily'] = pd.read_csv(macro_daily_path, parse_dates=['Date'])
            data['macro_daily']['Date'] = pd.to_datetime(data['macro_daily']['Date'], errors='coerce')
            logger.info(f"\n✓ Macro Daily: {data['macro_daily'].shape}")
            logger.info(f"  Date range: {data['macro_daily']['Date'].min().date()} to {data['macro_daily']['Date'].max().date()}")
            logger.info(f"  Use: VAE training (daily with quarterly aggregates)")
        else:
            logger.warning(f"  ⚠️  macro_features_daily.csv not found")

        # Load macro_features_quarterly (for merging with company)
        if macro_quarterly_path.exists():
            data['macro_quarterly'] = pd.read_csv(macro_quarterly_path, parse_dates=['Date'])
            data['macro_quarterly']['Date'] = pd.to_datetime(data['macro_quarterly']['Date'], errors='coerce')
            logger.info(f"\n✓ Macro Quarterly: {data['macro_quarterly'].shape}")
            logger.info(f"  Date range: {data['macro_quarterly']['Date'].min().date()} to {data['macro_quarterly']['Date'].max().date()}")
            logger.info(f"  Use: Merge with company data")
        else:
            logger.warning(f"  ⚠️  macro_features_quarterly.csv not found")

        # Load company_features (quarterly)
        if company_path.exists():
            data['company'] = pd.read_csv(company_path, parse_dates=['Date'])
            data['company']['Date'] = pd.to_datetime(data['company']['Date'], errors='coerce')
            logger.info(f"\n✓ Company: {data['company'].shape} (QUARTERLY)")
            logger.info(f"  Companies: {data['company']['Company'].nunique()}")
            logger.info(f"  Date range: {data['company']['Date'].min().date()} to {data['company']['Date'].max().date()}")
        else:
            raise FileNotFoundError(f"{company_path} not found. Run Step 2 first.")

        return data

    # ========== RESAMPLE TO QUARTERLY ==========

    def resample_to_quarterly(self, df: pd.DataFrame, name: str) -> pd.DataFrame:
        """
        Resample daily frequency to quarterly (quarter-end).
        
        Aggregation rules:
        - Price/index: last value of quarter
        - Volume: sum over quarter (if exists)
        - Returns/changes: mean over quarter
        - Volatility: mean over quarter
        """
        logger.info(f"\n   Resampling {name} to quarterly...")
        
        df = df.copy()
        df = df.set_index('Date').sort_index()
        
        # Identify numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # Define aggregation rules
        agg_dict = {}
        
        for col in numeric_cols:
            col_lower = col.lower()
            
            # Volume should be summed
            if 'volume' in col_lower:
                agg_dict[col] = 'sum'
            # Returns, changes should be averaged
            elif any(x in col_lower for x in ['return', 'change', 'growth']):
                agg_dict[col] = 'mean'
            # Volatility should be averaged
            elif any(x in col_lower for x in ['volatility', 'std', 'var']):
                agg_dict[col] = 'mean'
            # Everything else (prices, indices, ratios) - take last
            else:
                agg_dict[col] = 'last'
        
        # Resample to quarter-end
        df_quarterly = df.resample('Q').agg(agg_dict)
        
        # Handle categorical columns
        cat_cols = df.select_dtypes(exclude=[np.number]).columns
        for col in cat_cols:
            df_quarterly[col] = df[col].resample('Q').last()
        
        df_quarterly = df_quarterly.reset_index()
        
        logger.info(f"   {name}: {len(df):,} rows → {len(df_quarterly):,} rows (quarterly)")
        
        return df_quarterly

    # ========== PIPELINE 1: MACRO + MARKET (QUARTERLY) ==========

    def merge_pipeline1_quarterly(self, fred_df: pd.DataFrame, market_df: pd.DataFrame) -> pd.DataFrame:
        """
        Pipeline 1a: Merge FRED + Market at QUARTERLY frequency for VAE.
        
        Steps:
        1. FRED is already quarterly (or mixed frequency) - resample to quarterly
        2. Market is daily - resample to quarterly
        3. Merge on Date (quarter-end dates)
        """
        logger.info("\n" + "="*80)
        logger.info("PIPELINE 1a: MACRO + MARKET (QUARTERLY FOR VAE)")
        logger.info("="*80)

        # Resample FRED to quarterly (in case it has mixed frequencies)
        logger.info("\n1. Resampling FRED to quarterly...")
        fred_quarterly = self.resample_to_quarterly(fred_df, "FRED")

        # Resample Market to quarterly
        logger.info("\n2. Resampling Market to quarterly...")
        market_quarterly = self.resample_to_quarterly(market_df, "Market")

        # Merge on Date
        logger.info("\n3. Merging FRED + Market on quarter-end dates...")
        
        merged = pd.merge(
            fred_quarterly,
            market_quarterly,
            on='Date',
            how='outer',
            suffixes=('_fred', '_market')
        )

        merged.sort_values('Date', inplace=True)
        merged.reset_index(drop=True, inplace=True)

        logger.info(f"\n✓ Pipeline 1a merged (QUARTERLY): {merged.shape}")
        logger.info(f"  Date range: {merged['Date'].min().date()} to {merged['Date'].max().date()}")
        logger.info(f"  Quarters: {len(merged)}")
        
        # Check missing values
        missing_pct = (merged.isna().sum() / len(merged)) * 100
        high_missing = missing_pct[missing_pct > 10]
        
        if len(high_missing) > 0:
            logger.warning(f"\n  ⚠️  {len(high_missing)} columns with >10% missing")
            logger.info(f"      This is EXPECTED with quarterly data")
        else:
            logger.info(f"\n  ✓ Missing data: {merged.isna().sum().sum() / merged.size * 100:.2f}%")
        
        return merged

    # ========== PIPELINE 1b: MACRO + MARKET (DAILY) - NEW! ==========

    def merge_pipeline1_daily(self, fred_df: pd.DataFrame, market_df: pd.DataFrame) -> pd.DataFrame:
        """
        Pipeline 1b: Merge FRED + Market at DAILY frequency for VAE.
        
        NEW: Creates daily version for VAE training.
        
        Steps:
        1. Market is daily - keep as-is
        2. FRED is mixed frequency - forward fill to daily (OK for macro data)
        3. Merge on Date
        """
        logger.info("\n" + "="*80)
        logger.info("PIPELINE 1b: MACRO + MARKET (DAILY FOR VAE)")
        logger.info("="*80)

        logger.info("\n1. Preparing daily FRED data...")
        
        # Set Date as index
        fred_daily = fred_df.copy()
        fred_daily = fred_daily.set_index('Date').sort_index()
        
        # Get date range from market (daily)
        market_daily = market_df.copy()
        market_daily = market_daily.set_index('Date').sort_index()
        
        start_date = max(fred_daily.index.min(), market_daily.index.min())
        end_date = min(fred_daily.index.max(), market_daily.index.max())
        
        # Create daily date range
        daily_dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Reindex FRED to daily and forward fill
        # NOTE: Forward fill is OK here for macro data (changes slowly)
        fred_daily = fred_daily.reindex(daily_dates)
        fred_daily = fred_daily.ffill()  # Forward fill macro indicators
        fred_daily = fred_daily.reset_index().rename(columns={'index': 'Date'})
        
        logger.info(f"   FRED: {len(fred_df):,} rows → {len(fred_daily):,} rows (daily)")

        logger.info("\n2. Market data already daily...")
        market_daily = market_daily.reset_index().rename(columns={'index': 'Date'})
        logger.info(f"   Market: {len(market_daily):,} rows (daily)")

        # Merge on Date
        logger.info("\n3. Merging FRED + Market on daily dates...")
        
        merged = pd.merge(
            fred_daily,
            market_daily,
            on='Date',
            how='inner',  # Keep only dates with both FRED and Market
            suffixes=('_fred', '_market')
        )

        merged.sort_values('Date', inplace=True)
        merged.reset_index(drop=True, inplace=True)

        logger.info(f"\n✓ Pipeline 1b merged (DAILY): {merged.shape}")
        logger.info(f"  Date range: {merged['Date'].min().date()} to {merged['Date'].max().date()}")
        logger.info(f"  Days: {len(merged):,}")
        logger.info(f"  Missing: {merged.isna().sum().sum() / merged.size * 100:.2f}%")
        
        return merged

    # ========== PIPELINE 2: MACRO + MARKET + COMPANY ==========

    def merge_pipeline2(self, macro_quarterly_df: pd.DataFrame,
                       company_df: pd.DataFrame) -> pd.DataFrame:
        """
        Pipeline 2: Merge quarterly macro + company data for prediction.
        
        SIMPLIFIED: Both are already quarterly with features.
        Just need to merge on Date using merge_asof for alignment.
        """
        logger.info("\n" + "="*80)
        logger.info("PIPELINE 2: MACRO + COMPANY (FOR PREDICTION)")
        logger.info("="*80)

        logger.info(f"\n  Macro quarterly: {macro_quarterly_df.shape}")
        logger.info(f"  Company quarterly: {company_df.shape}")
        logger.info(f"  Companies: {company_df['Company'].nunique()}")

        # Use merge_asof for date alignment
        logger.info("\n  Using merge_asof for date alignment...")
        
        # Sort both by date
        company_sorted = company_df.sort_values('Date').copy()
        macro_sorted = macro_quarterly_df.sort_values('Date').copy()
        
        # Group by company and merge
        merged_parts = []
        
        for company in company_sorted['Company'].unique():
            company_data = company_sorted[company_sorted['Company'] == company].copy()
            
            # Merge asof: for each company date, find nearest past macro date
            company_merged = pd.merge_asof(
                company_data,
                macro_sorted,
                on='Date',
                direction='backward',
                tolerance=pd.Timedelta(days=100),
                suffixes=('', '_macro')
            )
            
            merged_parts.append(company_merged)
        
        merged = pd.concat(merged_parts, ignore_index=True)
        merged.sort_values(['Company', 'Date'], inplace=True)

        logger.info(f"\n✓ Pipeline 2 merged: {merged.shape}")
        logger.info(f"  Companies: {merged['Company'].nunique()}")
        logger.info(f"  Date range: {merged['Date'].min().date()} to {merged['Date'].max().date()}")

        # Check if macro columns merged
        macro_cols = [col for col in macro_sorted.columns if col not in ['Date', 'Quarter']]
        macro_present = sum(1 for col in macro_cols if col in merged.columns)
        
        logger.info(f"\n  Macro columns merged: {macro_present}/{len(macro_cols)}")
        
        if macro_present >= len(macro_cols) * 0.8:
            logger.info(f"  ✓ Macro data successfully merged")
        else:
            logger.warning(f"  ⚠️  Only {macro_present}/{len(macro_cols)} macro columns merged")

        # Check missing
        missing_pct = (merged.isna().sum().sum() / merged.size) * 100
        logger.info(f"\n  Missing: {missing_pct:.2f}%")

        return merged

    # ========== SAVE FUNCTIONS ==========

    def save_merged_datasets(self, pipeline1_quarterly: pd.DataFrame, 
                            pipeline1_daily: pd.DataFrame,
                            pipeline2_df: pd.DataFrame):
        """Save merged datasets (now includes daily macro for VAE)."""
        logger.info("\n" + "="*80)
        logger.info("SAVING MERGED DATASETS")
        logger.info("="*80)

        # Pipeline 1a: Quarterly (for quarterly VAE)
        pipeline1a_path = self.features_dir / 'macro_features_quarterly.csv'
        pipeline1_quarterly.to_csv(pipeline1a_path, index=False)
        
        logger.info(f"\n✓ Pipeline 1a (VAE - Quarterly):")
        logger.info(f"  Path:  {pipeline1a_path}")
        logger.info(f"  Shape: {pipeline1_quarterly.shape}")
        logger.info(f"  Size:  {pipeline1a_path.stat().st_size / (1024*1024):.2f} MB")
        logger.info(f"  Use:   Train VAE on quarterly scenarios")

        # Pipeline 1b: Daily (for daily VAE) - NEW!
        pipeline1b_path = self.features_dir / 'macro_features_daily.csv'
        pipeline1_daily.to_csv(pipeline1b_path, index=False)
        
        logger.info(f"\n✓ Pipeline 1b (VAE - Daily):")
        logger.info(f"  Path:  {pipeline1b_path}")
        logger.info(f"  Shape: {pipeline1_daily.shape}")
        logger.info(f"  Size:  {pipeline1b_path.stat().st_size / (1024*1024):.2f} MB")
        logger.info(f"  Use:   Train VAE on daily scenarios (MORE DATA)")

        # Pipeline 2: Quarterly (for prediction)
        pipeline2_path = self.features_dir / 'merged_features.csv'
        pipeline2_df.to_csv(pipeline2_path, index=False)
        
        logger.info(f"\n✓ Pipeline 2 (Prediction - Quarterly):")
        logger.info(f"  Path:  {pipeline2_path}")
        logger.info(f"  Shape: {pipeline2_df.shape}")
        logger.info(f"  Size:  {pipeline2_path.stat().st_size / (1024*1024):.2f} MB")
        logger.info(f"  Use:   Train XGBoost/LSTM for outcome prediction")

        # Save column lists
        logger.info(f"\n  Saving column lists...")
        
        with open(self.features_dir / 'pipeline1_quarterly_columns.txt', 'w') as f:
            f.write("PIPELINE 1a (VAE - Quarterly) - COLUMNS\n")
            f.write("="*80 + "\n\n")
            for i, col in enumerate(sorted(pipeline1_quarterly.columns), 1):
                f.write(f"{i:3d}. {col}\n")

        with open(self.features_dir / 'pipeline1_daily_columns.txt', 'w') as f:
            f.write("PIPELINE 1b (VAE - Daily) - COLUMNS\n")
            f.write("="*80 + "\n\n")
            for i, col in enumerate(sorted(pipeline1_daily.columns), 1):
                f.write(f"{i:3d}. {col}\n")

        with open(self.features_dir / 'pipeline2_columns.txt', 'w') as f:
            f.write("PIPELINE 2 (PREDICTION - Quarterly) - COLUMNS\n")
            f.write("="*80 + "\n\n")
            for i, col in enumerate(sorted(pipeline2_df.columns), 1):
                f.write(f"{i:3d}. {col}\n")

        logger.info(f"  ✓ Column lists saved")

    # ========== GENERATE REPORT ==========

    def generate_merge_report(self, pipeline1_quarterly: pd.DataFrame,
                             pipeline1_daily: pd.DataFrame,
                             pipeline2_df: pd.DataFrame):
        """Generate merge quality report."""
        logger.info(f"\n  Generating merge report...")
        
        report_data = []

        # Pipeline 1a stats (quarterly)
        report_data.append({
            'Pipeline': 'Pipeline 1a (VAE - Quarterly)',
            'Dataset': 'macro_features_quarterly.csv',
            'Input': 'FRED + Market',
            'Frequency': 'Quarterly',
            'Rows': len(pipeline1_quarterly),
            'Columns': len(pipeline1_quarterly.columns),
            'Missing_Pct': round((pipeline1_quarterly.isna().sum().sum() / pipeline1_quarterly.size) * 100, 2),
            'Date_Range': f"{pipeline1_quarterly['Date'].min().date()} to {pipeline1_quarterly['Date'].max().date()}"
        })

        # Pipeline 1b stats (daily) - NEW!
        report_data.append({
            'Pipeline': 'Pipeline 1b (VAE - Daily)',
            'Dataset': 'macro_features_daily.csv',
            'Input': 'FRED + Market',
            'Frequency': 'Daily',
            'Rows': len(pipeline1_daily),
            'Columns': len(pipeline1_daily.columns),
            'Missing_Pct': round((pipeline1_daily.isna().sum().sum() / pipeline1_daily.size) * 100, 2),
            'Date_Range': f"{pipeline1_daily['Date'].min().date()} to {pipeline1_daily['Date'].max().date()}"
        })

        # Pipeline 2 stats
        report_data.append({
            'Pipeline': 'Pipeline 2 (Prediction - Quarterly)',
            'Dataset': 'merged_features.csv',
            'Input': 'FRED + Market + Company',
            'Frequency': 'Quarterly',
            'Rows': len(pipeline2_df),
            'Columns': len(pipeline2_df.columns),
            'Missing_Pct': round((pipeline2_df.isna().sum().sum() / pipeline2_df.size) * 100, 2),
            'Date_Range': f"{pipeline2_df['Date'].min().date()} to {pipeline2_df['Date'].max().date()}"
        })

        report_df = pd.DataFrame(report_data)
        report_path = self.reports_dir / 'step3_merge_report.csv'
        report_df.to_csv(report_path, index=False)
        
        logger.info(f"  ✓ Report saved: {report_path}")
        
        return report_df

    # ========== MAIN PIPELINE ==========

    def run_merging_pipeline(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Execute merging pipeline.
        
        SIMPLIFIED: Step 2 already created the files we need.
        Just pass them through with proper naming.
        """
        logger.info("\n" + "="*80)
        logger.info("STEP 3: DATA MERGING (SIMPLIFIED)")
        logger.info("="*80)
        logger.info("\nStep 2 already created:")
        logger.info("  ✓ macro_features_daily.csv (for VAE)")
        logger.info("  ✓ macro_features_quarterly.csv (quarterly aggregates)")
        logger.info("\nStep 3 will:")
        logger.info("  1. Load macro_quarterly + company")
        logger.info("  2. Merge them → merged_features.csv")
        logger.info("  3. Copy macro_daily → macro_features_clean.csv (for VAE)")
        logger.info("="*80)

        overall_start = time.time()

        # Load data
        data = self.load_feature_datasets()

        if 'macro_daily' not in data:
            logger.error("❌ macro_features_daily.csv not found!")
            logger.error("   Run Step 2 first to create this file")
            raise FileNotFoundError("macro_features_daily.csv missing")

        if 'macro_quarterly' not in data:
            logger.error("❌ macro_features_quarterly.csv not found!")
            logger.error("   Run Step 2 first to create this file")
            raise FileNotFoundError("macro_features_quarterly.csv missing")

        # === Pipeline 1: Macro Daily (for VAE) - Just copy ===
        logger.info("\n" + "="*80)
        logger.info("PIPELINE 1: MACRO DAILY (FOR VAE)")
        logger.info("="*80)
        logger.info("\n  Macro daily already has all features from Step 2")
        logger.info("  Just copying to final location...")
        
        macro_daily = data['macro_daily'].copy()
        
        logger.info(f"\n✓ Pipeline 1 ready: {macro_daily.shape}")
        logger.info(f"  Date range: {macro_daily['Date'].min().date()} to {macro_daily['Date'].max().date()}")
        logger.info(f"  Rows: {len(macro_daily):,} (daily)")
        
        # === Pipeline 2: Macro Quarterly + Company ===
        logger.info("\n" + "="*80)
        logger.info("PIPELINE 2: MACRO QUARTERLY + COMPANY (FOR PREDICTION)")
        logger.info("="*80)
        
        pipeline2_merged = self.merge_pipeline2(data['macro_quarterly'], data['company'])

        # === Save ===
        logger.info("\n" + "="*80)
        logger.info("SAVING MERGED DATASETS")
        logger.info("="*80)

        # Save macro_daily as macro_features_clean.csv (for VAE)
        macro_output = self.features_dir / 'macro_features_clean.csv'
        macro_daily.to_csv(macro_output, index=False)
        logger.info(f"\n✓ Saved: {macro_output}")
        logger.info(f"  Shape: {macro_daily.shape}")
        logger.info(f"  Use: VAE training (daily with quarterly aggregates)")

        # Save merged_features.csv (for prediction)
        merged_output = self.features_dir / 'merged_features.csv'
        pipeline2_merged.to_csv(merged_output, index=False)
        logger.info(f"\n✓ Saved: {merged_output}")
        logger.info(f"  Shape: {pipeline2_merged.shape}")
        logger.info(f"  Use: XGBoost/LSTM training")

        elapsed = time.time() - overall_start

        # === SUMMARY ===
        logger.info("\n" + "="*80)
        logger.info("MERGING COMPLETE")
        logger.info("="*80)

        logger.info(f"\n📊 FINAL OUTPUT FILES:")
        logger.info(f"  1. macro_features_clean.csv")
        logger.info(f"     - Daily FRED + Market with quarterly aggregates")
        logger.info(f"     - Shape: {macro_daily.shape}")
        logger.info(f"     - Use: VAE training")
        
        logger.info(f"\n  2. merged_features.csv")
        logger.info(f"     - Quarterly company + macro data")
        logger.info(f"     - Shape: {pipeline2_merged.shape}")
        logger.info(f"     - Use: Prediction models")

        logger.info(f"\n⏱️  Total Time: {elapsed:.1f}s")
        logger.info("="*80)

        logger.info("\n✅ Step 3 Complete!")
        logger.info("\n➡️  Next: python step4_post_merge_cleaning.py")

        return macro_daily, pipeline2_merged


def main():
    """Execute Step 3: Data Merging."""

    merger = QuarterlyDataMerger(features_dir="data/features")

    try:
        macro_daily, merged_quarterly = merger.run_merging_pipeline()

        # === SHOW SAMPLE DATA ===
        logger.info("\n" + "="*80)
        logger.info("SAMPLE DATA PREVIEW")
        logger.info("="*80)

        logger.info("\n1. macro_features_clean.csv (for VAE) - First 5 rows:")
        display_cols = ['Date', 'GDP', 'VIX', 'vix_mean_q', 'sp500_q_return']
        available_cols = [col for col in display_cols if col in macro_daily.columns]
        if available_cols:
            print(macro_daily[available_cols].head().to_string(index=False))

        logger.info("\n2. merged_features.csv (for Prediction) - First 5 rows:")
        display_cols = ['Date', 'Company', 'Stock_Price', 'Revenue', 'net_margin_q', 'vix_mean_q']
        available_cols = [col for col in display_cols if col in merged_quarterly.columns]
        if available_cols:
            print(merged_quarterly[available_cols].head().to_string(index=False))

        logger.info("\n" + "="*80)
        logger.info("✅ STEP 3 SUCCESSFULLY COMPLETED")
        logger.info("="*80)

        return macro_daily, merged_quarterly

    except FileNotFoundError as e:
        logger.error(f"\n❌ ERROR: {e}")
        logger.error("\nMake sure you've run Step 2 first!")
        logger.error("  Run: python step2_feature_engineering.py")
        return None, None
    except Exception as e:
        logger.error(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return None, None


if __name__ == "__main__":
    merged_data = main()