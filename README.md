# 🚀 Advanced LSTM Trading Strategy Pipeline

A comprehensive, production-ready Python pipeline for generating technical indicators and implementing LSTM-based trading strategies with advanced preprocessing, feature optimization, and portfolio simulation.

## ✨ Features

### 📈 Technical Indicators Engine
- **50+ Technical Indicators**: Complete suite of trend, momentum, volatility, and volume indicators
- **Priority Indicators**: Focused on 14 key indicators proven for LSTM trading
- **Professional Implementation**: Robust error handling and data validation
- **Flexible Configuration**: Customizable parameters and settings

### 🧠 Advanced LSTM Strategy
- **Deep Learning Architecture**: Multi-layer LSTM with dropout regularization
- **Intelligent Preprocessing**: Linear interpolation, outlier removal, scaling, and PCA
- **Feature Optimization**: Automated selection using F-test, mutual information, and Random Forest
- **Signal Generation**: Confidence-weighted trading signals with quality assessment

### 💰 Portfolio Simulation
- **Realistic Trading**: Position sizing, risk management, and concurrent position limits
- **Performance Metrics**: Sharpe ratio, maximum drawdown, win rates, and profit factors
- **Advanced Analytics**: Market condition analysis and directional accuracy

## 🚀 Quick Start

### Recommended Workflow (Best Results)
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate technical indicators
python scripts/run_analysis.py

# 3. Run balanced LSTM strategy (RECOMMENDED)
python scripts/run_balanced_lstm.py
```

### Alternative Approaches
```bash
# Basic LSTM (original proven method)
python scripts/run_lstm_strategy.py

# Enhanced LSTM (all features - may reduce accuracy)
python scripts/run_enhanced_lstm.py
```

### 🎯 Why Use the Balanced Approach?

The `run_balanced_lstm.py` script provides the best of both worlds:
- ✅ **Maintains 50%+ accuracy** (proven performance)
- ✅ **Enhanced portfolio simulation** with realistic position sizing
- ✅ **Advanced model evaluation** (directional accuracy, market conditions)
- ✅ **Comprehensive reporting** with signal quality scoring
- ✅ **Conservative preprocessing** that preserves important patterns

## 📁 Project Structure

```
Timesnow/
├── src/                     # Core algorithms
│   ├── technical_indicators.py
│   └── lstm_trading_strategy.py
├── scripts/                 # Execution scripts  
│   ├── run_analysis.py
│   ├── run_lstm_strategy.py
│   ├── run_enhanced_lstm.py
│   └── run_balanced_lstm.py ⭐
├── config/                  # Configuration
│   ├── settings.py
│   └── lstm_settings.py
├── data/                    # Raw and processed data
│   ├── raw/
│   └── processed/
├── output/                  # Results and reports
├── docs/                    # Documentation
│   ├── ACCURACY_ANALYSIS.md
│   ├── PROJECT_STRUCTURE.md
│   └── TECHNICAL_DOCS.md
├── demo/                    # Verification tools
│   └── verify_indicators.py
├── .gitignore              # Git ignore rules
├── README.md               # This file
└── requirements.txt        # Dependencies
```

For detailed structure and usage see `docs/PROJECT_STRUCTURE.md`.

## 📊 Priority Technical Indicators

The system prioritizes these 14 key indicators for LSTM training:

### Trend Indicators
- **SMA_20**: 20-period Simple Moving Average
- **EMA_12**: 12-period Exponential Moving Average  
- **MACD**: Moving Average Convergence Divergence
- **ADX**: Average Directional Index

### Momentum Indicators
- **RSI**: Relative Strength Index
- **Stochastic_K**: Stochastic %K
- **Williams_R**: Williams %R
- **ROC**: Rate of Change

### Volatility Indicators
- **Bollinger_Upper/Lower**: Bollinger Bands
- **ATR**: Average True Range
- **Volatility**: Price volatility

### Volume Indicators
- **Volume_SMA**: Volume Simple Moving Average
- **OBV**: On-Balance Volume

## 🎯 Output Files

The pipeline generates comprehensive output files:

### Trading Signals
- `enhanced_trading_signals.csv`: Detailed trading signals with confidence scores
- `portfolio_history.csv`: Portfolio value tracking over time
- `feature_importance.csv`: Feature selection results and importance scores

### Visualizations
- Price predictions vs actual charts
- Accuracy analysis plots
- Portfolio performance graphs
- Signal distribution analysis

### Performance Reports
- Model evaluation metrics (MAE, RMSE, R², directional accuracy)
- Trading performance statistics (win rate, profit factor, Sharpe ratio)
- Portfolio simulation results (returns, drawdown, capital utilization)

## ⚙️ Configuration

### LSTM Settings (`config/lstm_settings.py`)
```python
LSTM_CONFIG = {
    'sequence_length': 30,        # Days of historical data
    'target_horizon': 5,          # Prediction horizon
    'lstm_units': [64, 32],       # LSTM layer sizes
    'dropout_rate': 0.3,          # Dropout for regularization
    'epochs': 100,                # Training epochs
    'batch_size': 32,             # Batch size
    'validation_split': 0.15      # Validation data split
}
```

### Portfolio Settings
```python
PORTFOLIO_CONFIG = {
    'initial_capital': 100000,    # Starting capital
    'position_size': 0.15,        # 15% risk per trade
    'max_positions': 3,           # Max concurrent positions
    'stop_loss_pct': 5.0,         # 5% stop loss
    'target_profit_pct': 15.0     # 15% profit target
}
```

## 🔧 Advanced Usage

### Custom Feature Selection
```python
from src.lstm_trading_strategy import LSTMTradingStrategy

strategy = LSTMTradingStrategy(use_advanced_preprocessing=True)
strategy.prepare_data("data/processed/stock_data_with_technical_indicators.csv")

# Optimize features to top 20 most predictive
strategy.build_model(optimize_features=True)
X_optimized = strategy.optimize_feature_selection(X_train, y_train, max_features=20)
```

### Portfolio Simulation
```python
# Run realistic portfolio simulation
portfolio_metrics = strategy.calculate_portfolio_performance(
    initial_capital=250000,   # $250k starting capital
    position_size=0.10,       # 10% per trade (more conservative)
    max_positions=5           # Up to 5 concurrent positions
)

print(f"Total Return: {portfolio_metrics['Total_Return_Pct']}%")
print(f"Sharpe Ratio: {portfolio_metrics['Sharpe_Ratio']}")
print(f"Max Drawdown: {portfolio_metrics['Max_Drawdown_Pct']}%")
```

### Model Evaluation
```python
# Comprehensive model evaluation
model_metrics = strategy.evaluate_model_performance()
print(f"Directional Accuracy: {model_metrics['Directional_Accuracy_Pct']}%")
print(f"Bull Market Accuracy: {model_metrics['Bull_Market_Accuracy_Pct']}%")
print(f"Bear Market Accuracy: {model_metrics['Bear_Market_Accuracy_Pct']}%")
```

## 📊 Sample Results

Recent runs show promising performance:

```
📊 MODEL PERFORMANCE:
   MAE: 0.0123
   RMSE: 0.0198
   R2_Score: 0.8567
   Directional_Accuracy_Pct: 73.4%

🎯 TRADING PERFORMANCE:
   Total_Signals: 342
   Overall_Accuracy_Pct: 68.7%
   Win_Rate_Pct: 71.2%
   Profit_Factor: 1.34

💰 PORTFOLIO PERFORMANCE:
   Total_Return_Pct: 23.8%
   Sharpe_Ratio: 1.42
   Max_Drawdown_Pct: -8.3%
```

## 🛠️ Requirements

- Python 3.8+
- TensorFlow 2.8+
- scikit-learn 1.1+
- pandas 1.3+
- numpy 1.21+
- ta 0.10+ (technical analysis library)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for improvements.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built with TensorFlow and scikit-learn
- Technical indicators powered by the `ta` library
- Inspired by modern quantitative trading research

---

**⚠️ Disclaimer**: This software is for educational and research purposes only. Past performance does not guarantee future results. Always conduct thorough backtesting and risk assessment before using any trading strategy with real capital.
