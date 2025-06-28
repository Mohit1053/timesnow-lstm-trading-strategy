# 🚀 Running LSTM Trading Strategy on Kaggle

This guide shows you how to run your LSTM trading strategy with 13 technical indicators on Kaggle.

## 📋 Quick Start Guide

### Option 1: Upload Single File (Recommended)
1. Go to [Kaggle Notebooks](https://www.kaggle.com/code)
2. Click "New Notebook"
3. Copy the entire content of `kaggle_lstm_trading_strategy.py`
4. Paste it into a new Kaggle notebook cell
5. Run the cell

### Option 2: Upload as Dataset
1. Create a new dataset on Kaggle
2. Upload `kaggle_lstm_trading_strategy.py`
3. In a new notebook, load and run the script:
```python
import sys
sys.path.append('/kaggle/input/your-dataset-name')
from kaggle_lstm_trading_strategy import run_kaggle_lstm_strategy

# Run the strategy
strategy, signals = run_kaggle_lstm_strategy()
```

## 📊 Using Your Own Data

### CSV Format Required
Your data should have these columns:
- `Date`: Date column (YYYY-MM-DD format)
- `Open`: Opening price
- `High`: Highest price of the day
- `Low`: Lowest price of the day  
- `Close`: Closing price
- `Volume`: Trading volume

### Loading Your Data
Replace the sample data loading section with:
```python
# Load your data from Kaggle input
df = pd.read_csv('/kaggle/input/your-dataset/your-file.csv')

# Ensure Date column is datetime
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

# Run the strategy
strategy, signals = run_kaggle_lstm_strategy(df)
```

## 🔧 Technical Indicators Included

The script automatically calculates these 13 indicators:

### 📈 Momentum Indicators
- **RSI**: Relative Strength Index (14-period)
- **ROC**: Rate of Change (12-period)
- **Stochastic**: %K and %D oscillators
- **TSI**: True Strength Index

### 📊 Volume Indicators  
- **OBV**: On Balance Volume
- **MFI**: Money Flow Index (14-period)
- **PVT**: Price Volume Trend

### 📉 Trend Indicators
- **TEMA**: Triple Exponential Moving Average
- **MACD**: Moving Average Convergence Divergence
- **KAMA**: Kaufman's Adaptive Moving Average

### 🌪️ Volatility Indicators
- **ATR**: Average True Range (14-period)
- **Bollinger Bands**: Position within bands
- **Ulcer Index**: Downside volatility measure

## ⚡ Kaggle Optimizations

The script includes several Kaggle-specific optimizations:

1. **Self-contained**: No external file dependencies
2. **Memory efficient**: Optimized for Kaggle's memory limits
3. **GPU ready**: Automatically uses GPU if available
4. **Fast execution**: Streamlined for quick results
5. **Clean output**: Minimal warnings and clean progress indicators

## 📈 Expected Output

The strategy will provide:

1. **Model Performance Metrics**:
   - MAE (Mean Absolute Error)
   - RMSE (Root Mean Square Error)
   - R² Score
   - Directional Accuracy

2. **Trading Signals**:
   - BUY/SELL recommendations
   - Confidence scores
   - Predicted vs actual returns
   - Signal accuracy statistics

3. **Visualizations**:
   - Price predictions vs actual
   - Training history
   - Signal performance by type
   - Return distributions

## 🎯 Customization Options

You can modify these parameters in the script:

```python
strategy = KaggleLSTMTradingStrategy(
    sequence_length=60,    # Days of history to use
    target_horizon=5,      # Days ahead to predict
    test_split=0.2         # 20% of data for testing
)

model = strategy.build_model(
    lstm_units=[100, 50],  # LSTM layer sizes
    dropout_rate=0.2       # Dropout for regularization
)

history = strategy.train_model(
    epochs=50,             # Training epochs
    batch_size=32,         # Batch size
    validation_split=0.2   # Validation split
)
```

## 📊 Sample Output Format

The strategy generates a signals DataFrame with these columns:

| Column | Description |
|--------|-------------|
| Day | Trading day number |
| Current_Price | Price when signal generated |
| Predicted_Price | LSTM prediction |
| Actual_Future_Price | Actual price after target horizon |
| Signal | BUY or SELL recommendation |
| Confidence | Signal confidence (0-10%) |
| Predicted_Change_% | Expected return |
| Actual_Change_% | Actual return achieved |
| Correct_Direction | Whether prediction was correct |

## 🚨 Important Notes

1. **Data Quality**: Ensure your data has no missing values or use the built-in cleaning
2. **Minimum Data**: Need at least 200-300 rows for meaningful results
3. **Computation Time**: Larger datasets may take 10-15 minutes to complete
4. **Memory Usage**: The script is optimized but very large datasets (>100MB) may hit limits

## 🏆 Best Practices for Kaggle

1. **Add Comments**: Explain your analysis for better notebook scores
2. **Markdown Cells**: Add explanatory text between code sections  
3. **Save Outputs**: Save important plots and results
4. **Make it Public**: Share your insights with the community

## 🔧 Troubleshooting

### Common Issues:

**"Feature not found" error**: 
- Check your CSV column names match the expected format
- Ensure you have OHLCV data

**"Insufficient data" error**:
- Need at least 100+ rows after the sequence length
- Check for and remove rows with missing values

**Memory error**:
- Reduce `sequence_length` or `batch_size`
- Use fewer epochs or smaller model

**Low accuracy**:
- Try different `sequence_length` values (30-90)
- Adjust `target_horizon` (3-10 days)
- Ensure data quality is good

## 📞 Support

For issues specific to this implementation, check:
1. Data format matches requirements
2. All required columns are present
3. No excessive missing values
4. Sufficient data volume (200+ rows recommended)

The script includes built-in error checking and will guide you to fix common issues.

## 🎉 Ready to Run!

Your Kaggle LSTM trading strategy is ready! The script will automatically:
- Load and validate your data
- Calculate all 13 technical indicators  
- Train the LSTM model with GPU acceleration
- Generate trading signals with confidence scores
- Create comprehensive visualizations
- Provide detailed performance metrics

Happy trading! 📈🚀
