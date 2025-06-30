# 🚀 Kaggle LSTM Trading Strategy - Quick Start Guide

## 📋 Overview
This directory contains Kaggle-optimized versions of the LSTM trading strategy that you can run directly in Kaggle notebooks or script environments.

## 📁 Files in this Directory

### 🎯 Main Execution Scripts
- **`kaggle_complete_strategy.py`** - 🌟 **RECOMMENDED** - Uses your existing codebase with processed data
- **`kaggle_existing_code.py`** - Alternative version for existing code integration
- **`kaggle_focused_lstm.py`** - Self-contained script with built-in indicators
- **`kaggle_notebook_cells.py`** - Notebook cells for step-by-step execution
- **`kaggle_lstm_trading_strategy.py`** - Original comprehensive version

### 📚 Documentation
- **`README_KAGGLE.md`** - Detailed Kaggle setup instructions
- **`QUICK_START.md`** - This file
- **`requirements_kaggle.txt`** - Required packages for Kaggle

### 🔧 Setup Files
- **`kaggle_notebook_template.py`** - Template for creating new notebooks

## 🚀 How to Run in Kaggle

### Option 1: Use Your Existing Code (Recommended for Your Setup)

1. **Upload your processed data to Kaggle**
   - Create a new Kaggle dataset with your processed CSV file (the one that already has indicators calculated)
   - Note the dataset URL/path

2. **Use the complete strategy script**
   - Copy the entire content of `kaggle_complete_strategy.py`
   - Paste it into a new Kaggle notebook cell
   - Update the `DATA_PATH` variable at the top:
   ```python
   DATA_PATH = '/kaggle/input/your-dataset-name/your-processed-file.csv'
   ```

3. **Run the cell**
   - The script uses your existing LSTMTradingStrategy class
   - It expects your processed data with indicators already calculated
   - Handles both 'close' and 'Close' column naming conventions
   - Automatically detects and selects company data from multi-company datasets
   - Applies PCA dimensionality reduction for datasets with >10 features
   - All outputs will be generated in `/kaggle/working/`

### Option 2: Single Script Execution (If you want built-in indicators)

1. **Upload your data to Kaggle**
   - Create a new Kaggle dataset with your `priceData5Year.csv` file
   - Or use an existing dataset

2. **Create a new Kaggle notebook**
   - Go to kaggle.com → Create → New Notebook
   - Choose "Notebook" type

3. **Copy and run the script**
   ```python
   # Copy the entire content of kaggle_focused_lstm.py
   # Paste it into your Kaggle notebook
   # Update the DATA_PATH variable to match your data location
   
   DATA_PATH = '/kaggle/input/your-dataset-name/priceData5Year.csv'
   ```

4. **Execute the script**
   - Run the cell containing the script
   - The script will automatically install required packages
   - All outputs will be generated in `/kaggle/working/`

### Option 2: Step-by-Step Notebook Execution

1. **Use the notebook cells template**
   - Copy sections from `kaggle_notebook_cells.py`
   - Create separate cells for each section (marked with `# CELL X:`)

2. **Configure settings**
   - Update the `CONFIG` dictionary in Cell 3
   - Set your data path and trading parameters

3. **Run cells sequentially**
   - Execute cells 1-10 in order
   - Monitor progress and outputs

## 📊 Expected Outputs

When you run the script successfully, you'll get:

### 📁 Generated Files
- `focused_lstm_results.csv` - Complete trading results with signals
- `focused_trading_log.csv` - Detailed log of all trades
- `focused_portfolio_summary.txt` - Performance summary report
- `focused_model_metrics.txt` - LSTM model evaluation metrics
- `focused_lstm_analysis.png` - Visualization charts

### 📈 Visualizations
- Price vs LSTM predictions
- Portfolio value over time  
- Trading signals (buy/sell points)
- Returns distribution histogram

### 💰 Performance Metrics
- Total return percentage
- Sharpe ratio
- Maximum drawdown
- Win rate and trade statistics

## ⚙️ Configuration Options

### 🔧 Key Settings to Modify

```python
CONFIG = {
    # Data settings
    'data_path': '/kaggle/input/your-dataset/file.csv',  # Update this!
    
    # LSTM model settings
    'sequence_length': 60,    # Days of history for prediction
    'lstm_units': 50,         # Neural network complexity
    'epochs': 30,             # Training iterations
    
    # Trading settings  
    'initial_capital': 100000,  # Starting money
    'transaction_cost': 0.001,  # 0.1% transaction fee
    'max_position_size': 0.95,  # Max 95% of capital per trade
}
```

## 🎯 The 13 Focused Indicators Used

### 📈 Momentum (4 indicators)
- **RSI** - Relative Strength Index
- **ROC** - Rate of Change  
- **Stochastic** - Stochastic Oscillator
- **TSI** - True Strength Index

### 📊 Volume (3 indicators)
- **OBV** - On Balance Volume
- **MFI** - Money Flow Index
- **PVT** - Price Volume Trend

### 📉 Trend (3 indicators)  
- **TEMA** - Triple Exponential Moving Average
- **MACD** - Moving Average Convergence Divergence
- **KAMA** - Kaufman's Adaptive Moving Average

### 🌪️ Volatility (3 indicators)
- **ATR** - Average True Range
- **Bollinger Bands** - Price bands and position
- **Ulcer Index** - Downside volatility measure

## 🐛 Troubleshooting

### Common Issues and Solutions

**1. Data Path Error**
```
Error: FileNotFoundError: No such file or directory
```
**Solution:** Update the `data_path` in CONFIG to match your actual Kaggle dataset path.

**2. Missing Columns Error**
```
Error: Missing columns: ['Open', 'High', 'Low', 'Close', 'Volume']
```
**Solution:** Ensure your CSV has these exact column names (case-sensitive).

**3. Package Installation Issues**
```
Error: No module named 'talib'
```
**Solution:** The script auto-installs packages. If it fails, manually run:
```python
!pip install TA-Lib tensorflow scikit-learn
```

**4. Memory Issues**
```
Error: ResourceExhaustedError: OOM when allocating tensor
```
**Solution:** Reduce `lstm_units` or `sequence_length` in CONFIG.

## 💡 Tips for Better Results

### 🎯 Parameter Optimization
- **More data = better predictions**: Use 2+ years of historical data
- **Adjust sequence_length**: Try 30-90 days based on your data frequency
- **Tune LSTM units**: Start with 50, increase for complex patterns

### ⚡ Performance Optimization
- **Reduce epochs** for faster execution (20-50 is usually enough)
- **Use validation split** to prevent overfitting
- **Monitor training loss** - stop if it plateaus

### 📊 Trading Strategy Tips
- **Conservative approach**: Lower `max_position_size` (0.5-0.8)
- **Risk management**: Set appropriate `stop_loss` and `take_profit`
- **Transaction costs**: Include realistic fees in `transaction_cost`

## 🔄 Next Steps After Running

1. **Analyze Results**
   - Check the generated PNG charts
   - Review the performance summary
   - Examine individual trades in the log

2. **Optimize Strategy**
   - Modify CONFIG parameters
   - Try different indicator combinations
   - Adjust trading rules

3. **Validate Results**
   - Test on different time periods
   - Compare with buy-and-hold strategy
   - Use walk-forward analysis

## 📞 Support

If you encounter issues:
1. Check the error message carefully
2. Verify your data format matches requirements
3. Try reducing model complexity (fewer units/epochs)
4. Ensure you have sufficient Kaggle compute resources

## 🎉 Success Indicators

You know it's working when you see:
- ✅ "All packages installed successfully!"
- ✅ "All indicators calculated successfully!"  
- ✅ "Training LSTM model..." with progress bars
- ✅ Generated files in `/kaggle/working/`
- ✅ Performance visualizations displayed

---

**Happy Trading! 📈🚀**
