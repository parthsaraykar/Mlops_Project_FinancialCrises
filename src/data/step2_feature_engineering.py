"""
STEP 2: COMPREHENSIVE FEATURE ENGINEERING (CLEAN VERSION)

Based on actual data structure:
- FRED: Daily data (macro indicators)
- Market: Daily data (VIX, SP500)
- Company Prices: Quarterly data (stock prices)
- Balance: Quarterly data
- Income: Quarterly data

Creates:
1. Macro features (daily FRED + Market with quarterly aggregates) → for VAE
2. Company features (quarterly fundamentals + market features) → for prediction

Output:
- macro_features_daily.csv (daily with quarterly aggregates, for VAE)
- company_features_quarterly.csv (quarterly, for prediction)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict
import time
import warnings

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


class CleanFeatureEngineer:
    """Clean feature engineering matching actual data structure."""

    def __init__(self, clean_dir: str = "data/clean", features_dir: str = "data/features"):
        self.clean_dir = Path(clean_dir)
        self.features_dir = Path(features_dir)
        self.features_dir.mkdir(parents=True, exist_ok=True)

    def load_cleaned_data(self) -> Dict[str, pd.DataFrame]:
        """Load all cleaned datasets."""
        logger.info("="*80)
        logger.info("LOADING CLEANED DATASETS")
        logger.info("="*80)

        data = {}

        for name, filename in [
            ('fred', 'fred_clean.csv'),
            ('market', 'market_clean.csv'),
            ('prices', 'company_prices_clean.csv'),
            ('balance', 'company_balance_clean.csv'),
            ('income', 'company_income_clean.csv')
        ]:
            path = self.clean_dir / filename
            if path.exists():
                df = pd.read_csv(path)
                # Handle Date parsing
                if 'Date' in df.columns:
                    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                data[name] = df
                logger.info(f"  ✓ {name}: {df.shape}")
            else:
                logger.warning(f"  ⚠️  {filename} not found")

        return data

    # ========== MACRO FEATURES (DAILY WITH QUARTERLY AGGREGATES) ==========

    def create_macro_daily_features(self, fred_df: pd.DataFrame, market_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create macro features from daily FRED + Market data.
        
        Output: Daily data with quarterly aggregate features.
        """
        logger.info("\n" + "="*80)
        logger.info("CREATING MACRO FEATURES (DAILY + QUARTERLY AGGREGATES)")
        logger.info("="*80)

        # Merge FRED + Market on Date
        logger.info("\n1. Merging FRED + Market (daily)...")
        
        macro = pd.merge(fred_df, market_df, on='Date', how='outer')
        macro = macro.sort_values('Date')
        
        logger.info(f"   Merged: {macro.shape}")
        logger.info(f"   Date range: {macro['Date'].min().date()} to {macro['Date'].max().date()}")
        
        # Add Quarter identifier
        macro['Quarter'] = macro['Date'].dt.to_period('Q')
        
        logger.info(f"\n2. Creating quarterly aggregates...")
        
        # Define columns to aggregate
        fred_cols = ['GDP', 'CPI', 'Unemployment_Rate', 'Federal_Funds_Rate', 
                     'Yield_Curve_Spread', 'Consumer_Confidence', 'Oil_Price',
                     'Corporate_Bond_Spread', 'TED_Spread', 'Treasury_10Y_Yield',
                     'High_Yield_Spread']
        
        market_cols = ['VIX', 'SP500_Close']
        
        # Calculate SP500 daily returns
        macro['sp500_ret_t'] = macro['SP500_Close'].pct_change()
        
        # Create quarterly aggregates
        quarterly_features = []
        
        for quarter in macro['Quarter'].unique():
            q_data = macro[macro['Quarter'] == quarter].copy()
            
            if len(q_data) < 2:
                continue
            
            q_features = {
                'Quarter': str(quarter),
                'quarter_start_date': q_data['Date'].iloc[0],
                'quarter_end_date': q_data['Date'].iloc[-1]
            }
            
            last20 = q_data.iloc[-20:] if len(q_data) >= 20 else q_data
            
            # === VIX FEATURES ===
            if 'VIX' in q_data.columns:
                q_features['vix_mean_q'] = q_data['VIX'].mean()
                q_features['vix_max_q'] = q_data['VIX'].max()
                q_features['vix_std_q'] = q_data['VIX'].std()
                q_features['vix_stress_days_q'] = (q_data['VIX'] > 30).sum()
                q_features['vix_last_month_mean_q'] = last20['VIX'].mean()
            
            # === SP500 FEATURES ===
            if 'SP500_Close' in q_data.columns:
                first_sp = q_data['SP500_Close'].iloc[0]
                last_sp = q_data['SP500_Close'].iloc[-1]
                
                q_features['sp500_q_return'] = ((last_sp - first_sp) / (first_sp + 1e-6)) * 100
                q_features['sp500_q_volatility'] = q_data['sp500_ret_t'].std() * 100
                
                # Max drawdown
                cummax = q_data['SP500_Close'].cummax()
                drawdown = ((q_data['SP500_Close'] / cummax) - 1) * 100
                q_features['sp500_q_max_drawdown'] = drawdown.min()
                
                q_features['sp500_big_drop_days_q'] = (q_data['sp500_ret_t'] < -0.02).sum()
            
            # === CREDIT STRESS ===
            for col, threshold in [
                ('Corporate_Bond_Spread', 3.0),
                ('High_Yield_Spread', 6.0),
                ('TED_Spread', 0.5)
            ]:
                if col in q_data.columns:
                    col_clean = col.lower().replace('_', '')
                    q_features[f'{col_clean}_mean_q'] = q_data[col].mean()
                    q_features[f'{col_clean}_max_q'] = q_data[col].max()
                    q_features[f'{col_clean}_std_q'] = q_data[col].std()
                    q_features[f'{col_clean}_stress_days_q'] = (q_data[col] > threshold).sum()
            
            # === RATES & CURVE ===
            if 'Treasury_10Y_Yield' in q_data.columns:
                q_features['t10y_mean_q'] = q_data['Treasury_10Y_Yield'].mean()
                q_features['t10y_change_q'] = q_data['Treasury_10Y_Yield'].iloc[-1] - q_data['Treasury_10Y_Yield'].iloc[0]
            
            if 'Federal_Funds_Rate' in q_data.columns:
                q_features['fed_funds_mean_q'] = q_data['Federal_Funds_Rate'].mean()
                q_features['fed_funds_change_q'] = q_data['Federal_Funds_Rate'].iloc[-1] - q_data['Federal_Funds_Rate'].iloc[0]
            
            if 'Yield_Curve_Spread' in q_data.columns:
                q_features['yc_spread_mean_q'] = q_data['Yield_Curve_Spread'].mean()
                q_features['yc_inversion_days_q'] = (q_data['Yield_Curve_Spread'] < 0).sum()
            
            # === MACRO (LAST VALUES) ===
            for col in ['GDP', 'CPI', 'Unemployment_Rate', 'Consumer_Confidence', 'Oil_Price']:
                if col in q_data.columns:
                    last_val = q_data[col].iloc[-1]
                    if pd.notna(last_val):
                        q_features[f'{col.lower()}_q'] = last_val
            
            quarterly_features.append(q_features)
        
        # Create quarterly aggregates DataFrame
        q_agg_df = pd.DataFrame(quarterly_features)
        
        logger.info(f"   ✓ Created {len(q_agg_df.columns) - 3} quarterly aggregate features")
        logger.info(f"   ✓ Quarters: {len(q_agg_df)}")
        
        # Merge quarterly aggregates back to daily data
        logger.info(f"\n3. Merging quarterly aggregates back to daily data...")
        
        macro['Quarter'] = macro['Quarter'].astype(str)
        macro_with_agg = pd.merge(macro, q_agg_df, on='Quarter', how='left')
        
        # Remove temporary columns
        macro_with_agg.drop(columns=['sp500_ret_t', 'quarter_start_date', 'quarter_end_date'], 
                           inplace=True, errors='ignore')
        
        logger.info(f"   ✓ Final macro features: {macro_with_agg.shape}")
        
        return macro_with_agg

    # ========== COMPANY FEATURES (QUARTERLY FUNDAMENTALS) ==========

    def create_company_quarterly_features(self, prices_df: pd.DataFrame,
                                         balance_df: pd.DataFrame,
                                         income_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create company features from quarterly data.
        
        Since prices are QUARTERLY (not daily), we create fundamental ratios
        and simple quarterly returns.
        """
        logger.info("\n" + "="*80)
        logger.info("CREATING COMPANY FEATURES (QUARTERLY)")
        logger.info("="*80)

        # === 1. MERGE ALL COMPANY DATA ===
        logger.info("\n1. Merging company datasets...")
        
        # Merge balance + income
        df = pd.merge(balance_df, income_df, on=['Date', 'Company'], how='outer', suffixes=('', '_inc'))
        
        # Remove duplicate columns
        dup_cols = [col for col in df.columns if col.endswith('_inc')]
        if dup_cols:
            df.drop(columns=dup_cols, inplace=True)
        
        # Merge with prices
        price_cols = ['Date', 'Company', 'Stock_Price', 'Volume']
        available_price_cols = [col for col in price_cols if col in prices_df.columns]
        df = pd.merge(df, prices_df[available_price_cols], on=['Date', 'Company'], how='outer')
        
        df = df.sort_values(['Company', 'Date'])
        
        logger.info(f"   Merged: {df.shape}")
        
        # === 2. FUNDAMENTAL FEATURES ===
        logger.info("\n2. Creating fundamental features...")
        
        # Scale / Size
        df['log_total_assets_q'] = np.log(df['Total_Assets'].fillna(1) + 1)
        df['log_revenue_q'] = np.log(df['Revenue'].fillna(0) + 1)
        
        # Leverage
        df['debt_to_assets_q'] = df['Total_Debt'] / (df['Total_Assets'] + 1e-6)
        df['long_term_debt_to_assets_q'] = df['Long_Term_Debt'] / (df['Total_Assets'] + 1e-6)
        df['short_term_debt_to_assets_q'] = df['Short_Term_Debt'] / (df['Total_Assets'] + 1e-6)
        df['short_term_debt_ratio_q'] = df['Short_Term_Debt'] / (df['Total_Debt'] + 1e-6)
        
        # Liquidity
        df['cash_to_assets_q'] = df['Cash'] / (df['Total_Assets'] + 1e-6)
        df['cash_to_current_liabilities_q'] = df['Cash'] / (df['Current_Liabilities'] + 1e-6)
        df['current_ratio_q'] = df['Current_Assets'] / (df['Current_Liabilities'] + 1e-6)
        
        # Profitability
        df['net_margin_q'] = df['Net_Income'] / (df['Revenue'] + 1e-6)
        df['gross_margin_q'] = df['Gross_Profit'] / (df['Revenue'] + 1e-6)
        df['operating_margin_q'] = df['Operating_Income'] / (df['Revenue'] + 1e-6)
        df['ebitda_margin_q'] = df['EBITDA'] / (df['Revenue'] + 1e-6)
        df['roa_q'] = df['Net_Income'] / (df['Total_Assets'] + 1e-6)
        df['roe_q'] = df['Net_Income'] / (df['Total_Equity'] + 1e-6)
        
        # Coverage
        df['ebitda_to_total_debt_q'] = df['EBITDA'] / (df['Total_Debt'] + 1e-6)
        
        logger.info(f"   ✓ Created 17 fundamental features")
        
        # === 3. QUARTERLY STOCK MARKET FEATURES ===
        logger.info("\n3. Creating quarterly stock market features...")
        
        # Simple quarterly return
        df['stock_q_return'] = df.groupby('Company')['Stock_Price'].pct_change() * 100
        
        # Volume features
        if 'Volume' in df.columns:
            df['stock_q_mean_volume'] = df['Volume']  # Already quarterly
        
        logger.info(f"   ✓ Created 2 stock market features")
        
        # === 4. TEMPORAL FEATURES ===
        logger.info("\n4. Creating temporal features (lags, deltas, rolling)...")
        
        # Key features for temporal analysis
        key_features = [
            'stock_q_return', 'net_margin_q', 'roa_q', 'roe_q',
            'debt_to_assets_q', 'log_revenue_q', 'log_total_assets_q'
        ]
        key_features = [f for f in key_features if f in df.columns]
        
        for col in key_features:
            # Lags (per company)
            df[f'{col}_t_1'] = df.groupby('Company')[col].shift(1)
            df[f'{col}_t_2'] = df.groupby('Company')[col].shift(2)
            df[f'{col}_t_4'] = df.groupby('Company')[col].shift(4)
            
            # Deltas
            df[f'Δ{col}_1q'] = df[col] - df[f'{col}_t_1']
            df[f'Δ{col}_4q'] = df[col] - df[f'{col}_t_4']
            
            # Rolling (4 quarters)
            df[f'rolling4q_avg_{col}'] = df.groupby('Company')[col].rolling(4, min_periods=1).mean().reset_index(0, drop=True)
            df[f'rolling4q_min_{col}'] = df.groupby('Company')[col].rolling(4, min_periods=1).min().reset_index(0, drop=True)
            df[f'rolling4q_max_{col}'] = df.groupby('Company')[col].rolling(4, min_periods=1).max().reset_index(0, drop=True)
        
        logger.info(f"   ✓ Created {len(key_features) * 8} temporal features")
        
        # Replace inf with NaN
        df = df.replace([np.inf, -np.inf], np.nan)
        
        logger.info(f"\n✓ Final company features: {df.shape}")
        
        return df

    def create_macro_quarterly_aggregates(self, macro_daily: pd.DataFrame) -> pd.DataFrame:
        """
        Create quarterly aggregates from daily macro data.
        
        Returns DataFrame with one row per quarter.
        """
        macro_daily = macro_daily.copy()
        macro_daily['Quarter'] = macro_daily['Date'].dt.to_period('Q')
        
        quarterly_features = []
        
        for quarter in macro_daily['Quarter'].unique():
            q_data = macro_daily[macro_daily['Quarter'] == quarter].copy()
            
            if len(q_data) < 2:
                continue
            
            q_features = {
                'Quarter': str(quarter),
                'Date': q_data['Date'].iloc[-1]  # Quarter-end date
            }
            
            last20 = q_data.iloc[-20:] if len(q_data) >= 20 else q_data
            
            # VIX features
            if 'VIX' in q_data.columns:
                q_features['vix_mean_q'] = q_data['VIX'].mean()
                q_features['vix_max_q'] = q_data['VIX'].max()
                q_features['vix_std_q'] = q_data['VIX'].std()
                q_features['vix_stress_days_q'] = (q_data['VIX'] > 30).sum()
                q_features['vix_last_month_mean_q'] = last20['VIX'].mean()
            
            # SP500 features
            if 'SP500_Close' in q_data.columns:
                first_sp = q_data['SP500_Close'].iloc[0]
                last_sp = q_data['SP500_Close'].iloc[-1]
                
                q_features['sp500_q_return'] = ((last_sp - first_sp) / (first_sp + 1e-6)) * 100
                
                sp_ret = q_data['SP500_Close'].pct_change()
                q_features['sp500_q_volatility'] = sp_ret.std() * 100
                
                # Max drawdown
                cummax = q_data['SP500_Close'].cummax()
                drawdown = ((q_data['SP500_Close'] / cummax) - 1) * 100
                q_features['sp500_q_max_drawdown'] = drawdown.min()
                
                q_features['sp500_big_drop_days_q'] = (sp_ret < -0.02).sum()
            
            # Credit stress features
            for col, threshold in [
                ('Corporate_Bond_Spread', 3.0),
                ('High_Yield_Spread', 6.0),
                ('TED_Spread', 0.5)
            ]:
                if col in q_data.columns:
                    col_short = col.lower().replace('_', '')[:10]  # Shorten name
                    q_features[f'{col_short}_mean_q'] = q_data[col].mean()
                    q_features[f'{col_short}_max_q'] = q_data[col].max()
                    q_features[f'{col_short}_std_q'] = q_data[col].std()
                    q_features[f'{col_short}_stress_days_q'] = (q_data[col] > threshold).sum()
            
            # Rates features
            if 'Treasury_10Y_Yield' in q_data.columns:
                q_features['t10y_mean_q'] = q_data['Treasury_10Y_Yield'].mean()
                q_features['t10y_change_q'] = q_data['Treasury_10Y_Yield'].iloc[-1] - q_data['Treasury_10Y_Yield'].iloc[0]
            
            if 'Federal_Funds_Rate' in q_data.columns:
                q_features['fedfunds_mean_q'] = q_data['Federal_Funds_Rate'].mean()
                q_features['fedfunds_change_q'] = q_data['Federal_Funds_Rate'].iloc[-1] - q_data['Federal_Funds_Rate'].iloc[0]
            
            if 'Yield_Curve_Spread' in q_data.columns:
                q_features['yc_spread_mean_q'] = q_data['Yield_Curve_Spread'].mean()
                q_features['yc_inversion_days_q'] = (q_data['Yield_Curve_Spread'] < 0).sum()
            
            # Macro last values
            for col in ['GDP', 'CPI', 'Unemployment_Rate', 'Consumer_Confidence', 'Oil_Price']:
                if col in q_data.columns:
                    last_val = q_data[col].iloc[-1]
                    if pd.notna(last_val):
                        q_features[f'{col.lower()}_q'] = last_val
            
            quarterly_features.append(q_features)
        
        q_df = pd.DataFrame(quarterly_features)
        q_df['Date'] = pd.to_datetime(q_df['Date'])
        
        return q_df

    # ========== MAIN PIPELINE ==========

    def run_feature_engineering(self) -> Dict[str, pd.DataFrame]:
        """Execute feature engineering."""
        logger.info("\n" + "="*80)
        logger.info("STEP 2: COMPREHENSIVE FEATURE ENGINEERING")
        logger.info("="*80)
        logger.info("\nStrategy:")
        logger.info("  1. Macro: Daily FRED + Market → quarterly aggregates")
        logger.info("  2. Company: Quarterly fundamentals + temporal features")
        logger.info("="*80)

        overall_start = time.time()

        data = self.load_cleaned_data()

        if not data or 'fred' not in data or 'market' not in data:
            logger.error("\n❌ Missing required data!")
            return {}

        # === MACRO FEATURES (DAILY + QUARTERLY AGGREGATES) ===
        logger.info("\n[1/2] MACRO FEATURES")
        
        macro_daily = self.create_macro_daily_features(data['fred'], data['market'])
        
        # Save daily macro with aggregates
        output_path = self.features_dir / 'macro_features_daily.csv'
        macro_daily.to_csv(output_path, index=False)
        logger.info(f"\n✓ Saved: {output_path}")
        logger.info(f"  Rows: {len(macro_daily):,} (daily)")
        logger.info(f"  Columns: {len(macro_daily.columns)}")
        
        # Also create quarterly version for compatibility
        macro_quarterly = self.create_macro_quarterly_aggregates(macro_daily)
        output_path = self.features_dir / 'macro_features_quarterly.csv'
        macro_quarterly.to_csv(output_path, index=False)
        logger.info(f"\n✓ Saved: {output_path}")
        logger.info(f"  Rows: {len(macro_quarterly):,} (quarterly)")

        # === COMPANY FEATURES (QUARTERLY) ===
        logger.info("\n[2/2] COMPANY FEATURES")
        
        if 'prices' in data and 'balance' in data and 'income' in data:
            company_features = self.create_company_quarterly_features(
                data['prices'],
                data['balance'],
                data['income']
            )
            
            output_path = self.features_dir / 'company_features.csv'
            company_features.to_csv(output_path, index=False)
            logger.info(f"\n✓ Saved: {output_path}")
            logger.info(f"  Rows: {len(company_features):,}")
            logger.info(f"  Columns: {len(company_features.columns)}")

        elapsed = time.time() - overall_start

        # === SUMMARY ===
        logger.info("\n" + "="*80)
        logger.info("FEATURE ENGINEERING COMPLETE")
        logger.info("="*80)

        logger.info(f"\n📊 OUTPUT FILES:")
        logger.info(f"  1. macro_features_daily.csv")
        logger.info(f"     - Daily FRED + Market data")
        logger.info(f"     - ~50 quarterly aggregate features")
        logger.info(f"     - Use: VAE training")
        
        logger.info(f"\n  2. macro_features_quarterly.csv")
        logger.info(f"     - Quarterly aggregates only")
        logger.info(f"     - ~50 features")
        logger.info(f"     - Use: Reference/validation")
        
        logger.info(f"\n  3. company_features.csv")
        logger.info(f"     - Quarterly company data")
        logger.info(f"     - ~17 fundamental + ~56 temporal features")
        logger.info(f"     - Use: Prediction models")

        logger.info(f"\n⏱️  Total Time: {elapsed:.1f}s")
        logger.info("="*80)

        logger.info("\n✅ Step 2 Complete!")
        logger.info("\n➡️  Next: python step3_data_merging.py")

        return {
            'macro_daily': macro_daily,
            'macro_quarterly': macro_quarterly,
            'company_features': company_features
        }


def main():
    """Execute Step 2."""

    engineer = CleanFeatureEngineer(clean_dir="data/clean", features_dir="data/features")

    try:
        features = engineer.run_feature_engineering()

        if not features:
            logger.error("\n❌ Feature engineering failed!")
            return None

        logger.info("\n✅ STEP 2 COMPLETE")
        
        return features

    except Exception as e:
        logger.error(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    features = main()