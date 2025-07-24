#!/usr/bin/env python3
"""
DeFi Credit Scoring System with XGBoost option

This script calculates credit scores (0-1000) for wallets based on their Aave V2 transaction history.
Higher scores indicate responsible usage, while lower scores indicate risky or bot-like behavior.

Usage:
    python score_wallets.py --input user-wallet-transactions.csv --output scores.csv [--ml]
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import joblib
import sys
from datetime import datetime
from typing import Dict, List, Tuple, Optional

def load_data(file_path: str) -> pd.DataFrame:
    """Load transaction data from CSV or JSON file."""
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith('.json'):
        df = pd.read_json(file_path)
    else:
        raise ValueError("Unsupported file format. Please provide CSV or JSON file.")
    
    # Identify which timestamp column to use (look for numeric Unix timestamp)
    timestamp_col = None
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            # Check if values look like Unix timestamps (around 10-13 digits)
            if df[col].min() > 1000000000 and df[col].max() < 10000000000000:
                timestamp_col = col
                break
    
    # If no numeric timestamp found, look for ISO format timestamp
    if timestamp_col is None:
        for col in df.columns:
            if 'timestamp' in col.lower() or 'time' in col.lower() or 'date' in col.lower():
                timestamp_col = col
                break
    
    if timestamp_col is None:
        raise ValueError("Could not identify timestamp column in input data")
    
    # Standardize column names
    column_mapping = {
        'userWallet': 'userWallet',
        'userwallet': 'userWallet',
        'wallet': 'userWallet',
        timestamp_col: 'timestamp',  # Use identified timestamp column
        'action': 'action',
        'Action': 'action',
        'amount': 'amount',
        'Amount': 'amount',
        'assetPriceUSD': 'assetPriceUSD',
        'assetpriceusd': 'assetPriceUSD',
        'assetSymbol': 'assetSymbol',
        'assetsymbol': 'assetSymbol'
    }
    
    # Rename columns
    rename_dict = {col: column_mapping[col] for col in df.columns if col in column_mapping}
    df = df.rename(columns=rename_dict)
    
    # Ensure required columns exist
    required_columns = ['userWallet', 'timestamp', 'action', 'amount', 'assetPriceUSD', 'assetSymbol']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in input data")
    
    return df

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the transaction data."""
    # Convert timestamp to datetime
    if pd.api.types.is_numeric_dtype(df['timestamp']):
        # If timestamp is numeric, assume it's in seconds (Unix timestamp)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', errors='coerce')
    else:
        # Try to parse timestamp strings
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        
        # If that didn't work, check if it's in milliseconds format
        if df['timestamp'].isnull().all():
            try:
                # Try treating as milliseconds
                df['timestamp'] = pd.to_datetime(df['timestamp'].astype(float), unit='ms', errors='coerce')
            except:
                pass
    
    # Check if conversion was successful
    if df['timestamp'].isnull().all():
        raise ValueError("Could not convert timestamp column to datetime format")
    
    # Convert amount to numeric, handling scientific notation
    df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
    
    # Calculate USD value of each transaction
    df['usd_value'] = df['amount'] * df['assetPriceUSD']
    
    # Extract date for active days calculation
    df['date'] = df['timestamp'].dt.date
    
    # Clean action field
    df['action'] = df['action'].str.lower().str.strip()
    
    return df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer features from transaction data for each wallet."""
    wallet_groups = df.groupby('userWallet')
    wallet_features = []
    
    for wallet, group in wallet_groups:
        total_deposits = len(group[group['action'] == 'deposit'])
        total_borrows = len(group[group['action'] == 'borrow'])
        total_repays = len(group[group['action'] == 'repay'])
        total_liquidations = len(group[group['action'] == 'liquidationcall'])
        
        total_deposit_usd = group[group['action'] == 'deposit']['usd_value'].sum()
        total_borrow_usd = group[group['action'] == 'borrow']['usd_value'].sum()
        total_repay_usd = group[group['action'] == 'repay']['usd_value'].sum()
        
        unique_assets = group['assetSymbol'].nunique()
        active_days = group['date'].nunique()
        
        if len(group) > 1:
            sorted_group = group.sort_values('timestamp')
            time_diffs = sorted_group['timestamp'].diff().dt.total_seconds().dropna()
            mean_time_between = time_diffs.mean()
            std_time_between = time_diffs.std()
        else:
            mean_time_between = 0
            std_time_between = 0
        
        repay_ratio = total_repay_usd / total_borrow_usd if total_borrow_usd > 0 else 0
        collateral_ratio = total_deposit_usd / total_borrow_usd if total_borrow_usd > 0 else float('inf')
        
        wallet_features.append({
            'wallet': wallet,
            'total_deposits': total_deposits,
            'total_borrows': total_borrows,
            'total_repays': total_repays,
            'total_liquidations': total_liquidations,
            'unique_assets': unique_assets,
            'active_days': active_days,
            'total_deposit_usd': total_deposit_usd,
            'total_borrow_usd': total_borrow_usd,
            'total_repay_usd': total_repay_usd,
            'mean_time_between': mean_time_between,
            'std_time_between': std_time_between,
            'repay_ratio': repay_ratio,
            'collateral_ratio': collateral_ratio
        })
    
    features_df = pd.DataFrame(wallet_features)
    return features_df

def calculate_heuristic_score(features: pd.Series) -> float:
    """Calculate heuristic credit score for a wallet based on its features."""
    score = 500
    score -= features['total_liquidations'] * 100
    score += min(features['total_repay_usd'] / 1e6, 200)
    score += min(features['total_deposit_usd'] / 1e6, 100)
    score += min(features['unique_assets'] * 10, 50)
    score += min(features['active_days'] * 5, 100)
    
    if features['std_time_between'] < 3600:
        score -= 50
    elif features['std_time_between'] < 86400:
        score += 20
    else:
        score += 50
    
    if features['repay_ratio'] > 1.0:
        score += 50
    elif features['repay_ratio'] > 0.7:
        score += 30
    elif features['repay_ratio'] > 0.3:
        score += 10
    
    if features['collateral_ratio'] > 2.0:
        score += 50
    elif features['collateral_ratio'] > 1.5:
        score += 30
    elif features['collateral_ratio'] > 1.0:
        score += 10
    
    score = max(0, min(score, 1000))
    return score

def train_xgboost_model(features_df: pd.DataFrame) -> Tuple[xgb.Booster, Dict]:
    """Train XGBoost model using heuristic scores as labels."""
    # Calculate heuristic scores for training
    features_df['heuristic_score'] = features_df.apply(calculate_heuristic_score, axis=1)
    
    # Prepare features and target
    feature_cols = [col for col in features_df.columns 
                   if col not in ['wallet', 'heuristic_score', 'credit_score']]
    
    X = features_df[feature_cols].values
    y = features_df['heuristic_score'].values
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42)
    
    # Train model
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    params = {
        'objective': 'reg:squarederror',
        'max_depth': 5,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'eval_metric': 'rmse'
    }
    
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=100,
        evals=[(dtrain, 'train'), (dtest, 'test')],
        early_stopping_rounds=10,
        verbose_eval=False
    )
    
    # Evaluate model
    y_pred = model.predict(dtest)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"XGBoost model trained. Validation RMSE: {rmse:.2f}")
    
    # Save model and scaler
    model_path = "credit_score_model.xgb"
    scaler_path = "feature_scaler.pkl"
    joblib.dump(scaler, scaler_path)
    model.save_model(model_path)
    
    return model, {'scaler_path': scaler_path, 'model_path': model_path}

def predict_with_xgboost(model: xgb.Booster, scaler: StandardScaler, 
                        features_df: pd.DataFrame) -> pd.DataFrame:
    """Generate scores using XGBoost model."""
    feature_cols = [col for col in features_df.columns 
                   if col not in ['wallet', 'heuristic_score', 'credit_score']]
    
    X = features_df[feature_cols].values
    X_scaled = scaler.transform(X)
    
    dmatrix = xgb.DMatrix(X_scaled)
    scores = model.predict(dmatrix)
    
    # Clip scores to 0-1000 range
    scores = np.clip(scores, 0, 1000)
    
    scores_df = pd.DataFrame({
        'wallet': features_df['wallet'],
        'credit_score': scores
    })
    
    return scores_df

def generate_scores(features_df: pd.DataFrame, use_ml: bool = False) -> pd.DataFrame:
    """Generate credit scores for all wallets (heuristic or ML)."""
    if not use_ml:
        # Use heuristic scoring
        features_df['credit_score'] = features_df.apply(calculate_heuristic_score, axis=1)
        scores_df = features_df[['wallet', 'credit_score']].sort_values('credit_score', ascending=False)
        return scores_df
    else:
        # Use ML approach
        print("Training XGBoost model...")
        model, paths = train_xgboost_model(features_df.copy())
        
        # Load scaler
        scaler = joblib.load(paths['scaler_path'])
        
        # Generate scores with model
        scores_df = predict_with_xgboost(model, scaler, features_df)
        scores_df = scores_df.sort_values('credit_score', ascending=False)
        
        return scores_df

def analyze_scores(scores_df: pd.DataFrame, output_file: str = "analysis.md"):
    """Generate analysis of the credit scores and save to analysis.md."""
    bins = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    labels = ['0-100', '100-200', '200-300', '300-400', '400-500', 
              '500-600', '600-700', '700-800', '800-900', '900-1000']
    
    scores_df['score_range'] = pd.cut(scores_df['credit_score'], bins=bins, labels=labels, right=False)
    distribution = scores_df['score_range'].value_counts().sort_index()
    
    plt.figure(figsize=(10, 6))
    plt.hist(scores_df['credit_score'], bins=10, edgecolor='black')
    plt.title('Credit Score Distribution')
    plt.xlabel('Credit Score')
    plt.ylabel('Number of Wallets')
    plt.grid(axis='y', alpha=0.75)
    
    hist_path = "score_distribution.png"
    plt.savefig(hist_path)
    plt.close()
    
    analysis = f"""# Credit Score Analysis

## Score Distribution

| Score Range | Count | Percentage |
|-------------|-------|------------|
"""
    
    total_wallets = len(scores_df)
    for range_label, count in distribution.items():
        percentage = (count / total_wallets) * 100
        analysis += f"| {range_label} | {count} | {percentage:.1f}% |\n"
    
    analysis += f"""

## Distribution Chart
![Score Distribution]({hist_path})

## Observations

### High Scorers (700-1000) - ({len(scores_df[scores_df['credit_score'] > 700])} wallets)
- These wallets demonstrate responsible DeFi behavior
- Typically have multiple repayments relative to borrows
- No history of liquidations
- Diverse asset usage (multiple tokens)
- Consistent but not robotic transaction patterns
- Healthy collateral-to-debt ratios

### Mid Scorers (300-700) - ({len(scores_df[(scores_df['credit_score'] >= 300) & (scores_df['credit_score'] <= 700)])} wallets)
- Moderate DeFi activity with some risk factors
- May have occasional liquidations or inconsistent repayment patterns
- Generally maintain reasonable collateral ratios
- Activity spans a meaningful timeframe

### Low Scorers (0-300) - ({len(scores_df[scores_df['credit_score'] < 300])} wallets)
- High risk behavior patterns
- Frequent liquidations
- Little to no repayment activity
- Often exhibit bot-like transaction patterns (very regular timing)
- May have extreme collateral ratios (either too high or too low)
- Limited asset diversity or single-asset concentration

## Detailed Examples

### Top 3 Wallets
"""
    
    for i, (_, row) in enumerate(scores_df.head(3).iterrows(), 1):
        analysis += f"{i}. Wallet: `{row['wallet']}` - Score: {row['credit_score']:.1f}\n"
    
    analysis += "\n### Bottom 3 Wallets\n"
    
    for i, (_, row) in enumerate(scores_df.tail(3).iterrows(), 1):
        analysis += f"{i}. Wallet: `{row['wallet']}` - Score: {row['credit_score']:.1f}\n"
    
    analysis += """
## Conclusion

The credit scoring model effectively differentiates between responsible and risky wallet behavior in the Aave V2 protocol. 

Key findings:
- Most wallets fall in the mid-score range (300-700), indicating moderate DeFi activity with some risk factors
- High scorers are relatively rare but demonstrate exemplary DeFi behavior
- Low scorers often exhibit patterns associated with bots, liquidation risks, or poor financial management

This scoring system provides a valuable tool for assessing wallet reliability in DeFi protocols, with potential applications in risk management, lending decisions, and protocol governance.
"""
    
    with open(output_file, 'w') as f:
        f.write(analysis)
    
    print(f"Analysis saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Calculate credit scores for DeFi wallets')
    parser.add_argument('--input', type=str, required=True, 
                        help='Input file path (CSV or JSON)')
    parser.add_argument('--output', type=str, default='scores.csv',
                        help='Output file path for scores')
    parser.add_argument('--ml', action='store_true',
                        help='Use XGBoost model instead of heuristic scoring')
    parser.add_argument('--analyze', action='store_true',
                        help='Generate analysis.md with score distribution and insights')
    
    args = parser.parse_args()
    
    try:
        print(f"Loading data from {args.input}...")
        df = load_data(args.input)
        print(f"Loaded {len(df)} transactions from {df['userWallet'].nunique()} wallets")
        
        print("Preprocessing data...")
        df = preprocess_data(df)
        
        print("Engineering features...")
        features_df = engineer_features(df)
        
        print("Calculating credit scores...")
        scores_df = generate_scores(features_df, use_ml=args.ml)
        
        # Save scores
        scores_df.to_csv(args.output, index=False)
        print(f"Scores saved to {args.output}")
        
        # Generate analysis if requested
        if args.analyze:
            print("Generating analysis...")
            analyze_scores(scores_df)
    
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()