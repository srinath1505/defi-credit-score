# DeFi Credit Scoring System

![DeFi Credit Score](https://img.shields.io/badge/DeFi-Credit_Score-green)
![Python](https://img.shields.io/badge/Python-3.7%2B-blue)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

## Overview

This project calculates credit scores (0-1000) for cryptocurrency wallets based on their transaction history in the Aave V2 DeFi protocol. Higher scores indicate responsible usage patterns, while lower scores reflect risky, bot-like, or exploitative behavior.

## Problem Statement

You are provided with raw, transaction-level data from the Aave V2 protocol. Each record corresponds to a wallet interacting with the protocol through actions such as `deposit`, `borrow`, `repay`, `redeemunderlying`, and `liquidationcall`.

The goal is to develop a robust system that assigns a credit score between 0 and 1000 to each wallet, based solely on historical transaction behavior.

## Methodology

### Feature Engineering

We extract the following key features from each wallet's transaction history:

- **Transaction Metrics**: Counts of deposits, borrows, repayments, and liquidations
- **USD Value Metrics**: Total USD value of deposits, borrows, and repayments
- **Diversity Metrics**: Number of unique assets interacted with
- **Activity Duration**: Number of days with transactions
- **Timing Patterns**: Mean and standard deviation of time between transactions
- **Financial Ratios**: Repayment ratio and collateral ratio

### Scoring Approach

We offer two scoring approaches:

1. **Heuristic-Based Scoring**:
   - Starts with a base score of 500
   - Penalizes liquidations heavily
   - Rewards repayments, deposits, and asset diversity
   - Considers transaction timing patterns
   - Rewards good repayment ratios and healthy collateralization

2. **Machine Learning (XGBoost)**:
   - Uses heuristic scores as training labels
   - Trains on engineered features
   - Provides more nuanced scoring based on patterns in the data
   - Can be retrained as more data becomes available

## Installation

```bash
# Clone the repository
git clone https://github.com/srinath1505/defi-credit-score
cd defi-credit-score

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/MacOS
venv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
python score_wallets.py --input user-wallet-transactions.csv --output scores.csv [--ml] [--analyze]
```

### Command Line Arguments

| Argument | Description |
|----------|-------------|
| `--input` | Path to input CSV/JSON file (required) |
| `--output` | Path to save scores (default: scores.csv) |
| `--ml` | Use XGBoost model instead of heuristic scoring |
| `--analyze` | Generate analysis.md with score distribution and insights |

## Output

The tool generates:
- `scores.csv`: CSV file containing wallet addresses and their credit scores
- `analysis.md` (if `--analyze` flag used): Detailed analysis of score distribution and wallet behavior patterns
- `score_distribution.png`: Histogram of score distribution
- `credit_score_model.xgb` and `feature_scaler.pkl` (if using ML): Trained model files

## Architecture

```
[Input Data] 
    ↓
[Data Loader & Preprocessor]
    ↓
[Feature Engineering]
    ↓
[Scoring Engine (Heuristic or ML)]
    ↓
[Output: Scores & Analysis]
```

## Example

```bash
# Basic usage with heuristic scoring
python score_wallets.py --input data/user-wallet-transactions.csv --output scores.csv --analyze

# Using ML approach
python score_wallets.py --input data/user-wallet-transactions.csv --output scores.csv --ml --analyze
```

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


## Contact

Project Maintainer - Srinath S - srinathselvakumar1505@gmail.com

Project Link: [https://github.com/srinath1505/defi-credit-score](https://github.com/srinath1505/defi-credit-score)
```

To create this as a downloadable file:

1. Copy all the text above
2. Open a text editor (like Notepad, VS Code, Sublime Text, etc.)
3. Paste the copied text
4. Save the file as `README.md` (make sure the extension is `.md`, not `.txt`)
5. The file is now ready to use in your GitHub repository

Alternatively, you can create this file directly in your project directory using the command line:

```bash
# On Windows
echo ^<paste the markdown content here^> > README.md

# On Mac/Linux
echo "<paste the markdown content here>" > README.md
```
