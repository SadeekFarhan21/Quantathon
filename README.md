# Quantitative Market Analysis Toolkit

A professional-grade system for market state analysis, prediction, and investment strategy evaluation. This toolkit uses advanced machine learning techniques to classify market conditions, predict future states, and backtest multiple investment strategies.

![Market States Visualization](./docs/images/market_states_example.png)

## Core Capabilities

### Market Classification

- **Bear Market**: Negative trend with significant drawdown
- **Bull Market**: Positive trend with sustained growth
- **Static Market**: Sideways movement or consolidation

### Predictive Models

- **Traditional ML**: Gradient boosting with feature engineering
- **Neural Networks**: LSTM and Attention-based models for time series
- **Ensemble Methods**: Combined models for robust prediction

### Strategy Backtesting

- **Buy-and-Hold**: Baseline strategy for comparison
- **Prediction-Based**: Asset allocation based on market predictions
- **Dynamic Allocation**: Adaptive allocation using probability signals
- **Combined Strategy**: Multi-factor approach with risk management
- **Anomaly-Aware**: Strategy that responds to detected market anomalies

## Getting Started

### Installation

```bash
# Clone the repository
git clone https://github.com/SadeekFarhan21/Quantathon.git
cd Quantathon

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

Run a complete market analysis with default settings:

```bash
python main.py --data data/market_data.xlsx --output results
```

### Custom Analysis Period

Analyze a specific market period:

```bash
python main.py --data data/market_data.xlsx \ --start_date 2018-01-01 --end_date 2022-12-31 \ --train_start_date 2008-01-01 --train_end_date 2017-12-31
```

### Advanced Neural Network Models

Use deep learning capabilities:

```bash
python main.py --data data/market_data.xlsx --advanced --model_type attention
```

## Input Data Format

The system expects an Excel file with the following structure:

### Price Sheet

- **Date**: Trading dates
- **SP500**: S&P 500 index values
- **BondRate**: Interest rates for bond alternatives

### Probability Sheet

- **Date**: Trading dates
- **PrDec**: Probability of significant decrease
- **PrInc**: Probability of significant increase

## Key Components

### Market State Classification

Identifies market regimes based on price trends, volatility, and drawdown characteristics. This classification forms the foundation for prediction and strategy development.

### Anomaly Detection

Identifies unusual market behavior through multiple methods:

- Isolation forest for outlier detection
- Statistical analysis of volatility spikes
- Price jump identification
- DBSCAN clustering for pattern detection

### Risk Analysis

Advanced risk metrics including:

- Value at Risk (VaR) calculations
- Expected Shortfall (ES)
- Maximum drawdown analysis
- Stress testing of extreme scenarios

### Performance Evaluation

Comprehensive metrics for strategy evaluation:

- Risk-adjusted returns (Sharpe, Sortino)
- Drawdown characteristics
- Win rate and recovery periods
- Comparative performance visualization

## Implementation Details

The codebase follows a modular architecture:

```
quant/
│
├── src/                # Core functionality
│   ├── data_loader.py  # Data import and preprocessing
│   ├── market_classifier.py # Market state identification
│   ├── prediction_model.py  # ML prediction models
│   ├── backtest.py     # Strategy implementation
│   ├── market_anomaly.py   # Anomaly detection
│   └── advanced_models.py  # Neural network models
│
├── scripts/            # Analysis utilities
│   ├── analyze_probabilities.py
│   └── bond_rate_analysis.py
│
├── main.py             # Main execution script
└── requirements.txt    # Dependencies
```

## System Architecture

The system is designed with a modular architecture to ensure flexibility and scalability. Here is an overview of how the components interact with each other:

1. **Data Loader**: Imports and preprocesses market data from various sources.
2. **Market Classifier**: Identifies market states (Bear, Bull, Static) based on historical data.
3. **Prediction Model**: Utilizes machine learning models to predict future market states.
4. **Backtesting Engine**: Simulates investment strategies based on historical data and model predictions.
5. **Anomaly Detection**: Identifies unusual market behaviors that may impact strategy performance.
6. **Risk Analysis**: Evaluates the risk associated with different strategies using advanced metrics.
7. **Performance Evaluation**: Assesses the performance of strategies using various financial metrics.

The following diagram illustrates the interaction between these components:

```
+------------------+       +------------------+       +------------------+
|   Data Loader    | ----> | Market Classifier| ----> | Prediction Model |
+------------------+       +------------------+       +------------------+
        |                        |                        |
        v                        v                        v
+------------------+       +------------------+       +------------------+
| Backtesting Engine| <----| Anomaly Detection| <----| Risk Analysis    |
+------------------+       +------------------+       +------------------+
        |                        |                        |
        v                        v                        v
+---------------------------------------------------------------+
|                    Performance Evaluation                     |
+---------------------------------------------------------------+
```

## Strategy Descriptions

### Buy and Hold

Simple benchmark strategy that invests in S&P 500 and holds for the entire period.

### Prediction-Based Strategy

Binary allocation model that invests 100% in stocks during predicted Bull markets and 100% in bonds during Bear or Static markets.

### Dynamic Allocation

Weighted allocation based on prediction probabilities, allowing for more nuanced positioning.

### Combined Strategy

Multi-signal approach using:

- Market prediction signals
- Trend indicators
- Volatility measures
- Probability metrics

### Anomaly-Aware Strategy

Adaptive strategy that:

1. Reduces equity exposure when anomalies are detected
2. Gradually returns to normal allocation as market stabilizes
3. Uses specialized risk management during volatile periods

## Examples

### Market State Classification

![Market States](./docs/images/market_classification.png)

### Strategy Performance

![Strategy Performance](./docs/images/strategy_performance.png)

### Risk Analysis

![Drawdown Analysis](./docs/images/drawdown_analysis.png)

## Advanced Usage

### Fine-Tuning Predictions

Adjust model hyperparameters:

```bash
python main.py --data data/market_data.xlsx --advanced \
    --model_type ensemble --train_start_date 2005-01-01
```

### Custom Model Integration

The system is designed for extensibility. You can add custom models by:

1. Creating a new model class in `src/custom_models.py`
2. Implementing the required `train()` and `predict()` methods
3. Importing and initializing your model in `main.py`

## Command-Line Arguments and Usage

The `main.py` script provides a flexible command-line interface to control the market analysis and backtesting pipeline. Here's a detailed breakdown of the available arguments:

### General Arguments

* `--data`: (Required) Path to the market data file (Excel format).
  * Example: `--data data/market_data.xlsx`
* `--output`: (Optional) Directory to save the results and analysis. Default: `results`.
  * Example: `--output my_analysis_results`
* `--verbose`: (Optional) Enable verbose output for detailed logging.
  * Example: `--verbose`
* `--enhanced`: (Optional) Enable enhanced backtesting strategies.
  * Example: `--enhanced`
* `--max_leverage`: (Optional) Maximum leverage allowed in enhanced strategies (as a multiplier of capital). Default: `1.5`.
  * Example: `--max_leverage 1.2`
* `--markov`: (Optional) Enable Markov chain prediction strategy.
  * Example: `--markov`

### Date Range Arguments

These arguments control the date ranges used for analysis and training. It's crucial to set these correctly to ensure meaningful results.

* `--start_date`: (Optional) Start date for the analysis period (YYYY-MM-DD). If not specified, the analysis will start from the beginning of the available data.
  * Example: `--start_date 2019-01-01`
* `--end_date`: (Optional) End date for the analysis period (YYYY-MM-DD). If not specified, the analysis will end at the end of the available data. Default: `2022-12-31`.
  * Example: `--end_date 2022-12-31`
* `--train_start_date`: (Optional) Start date for the training data (YYYY-MM-DD). If not specified, the training data will start from the beginning of the available data.
  * Example: `--train_start_date 2008-01-01`
* `--train_end_date`: (Optional) End date for the training data (YYYY-MM-DD). If not specified, the training data will end before the analysis period.
  * Example: `--train_end_date 2018-12-31`

**Important Notes on Date Ranges:**

* The `start_date` and `end_date` define the period over which the backtesting and performance analysis are conducted.
* The `train_start_date` and `train_end_date` define the period used to train the prediction model.
* The training period should generally precede the analysis period to avoid lookahead bias.
* If no training period is specified, the system will use a default approach: either all data before the analysis period or the first 80% of the available data.
* The system will automatically adjust the date ranges to fit within the available data. Warnings will be logged if the specified dates are outside the available range.

### Advanced Model Arguments

These arguments control the use of advanced neural network models.

* `--advanced`: (Optional) Enable the use of advanced PyTorch-based models.
  * Example: `--advanced`
* `--model_type`: (Optional) Type of advanced model to use. Choices: `attention`, `tcn`, `ensemble`. Default: `attention`.

## Example Commands

### Analyzing a Specific Period with a Trained Model

To analyze the market from January 1, 2019, to December 31, 2022, using a model trained on data up to December 31, 2018, run the following command:
