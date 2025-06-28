# 🚀 Quick Kaggle Setup Guide

## 3 Ways to Run Your LSTM Strategy on Kaggle

### 🎯 Method 1: Copy-Paste Single File (Fastest)
1. Go to [Kaggle Notebooks](https://www.kaggle.com/code)
2. Click "New Notebook" 
3. Copy entire content from `kaggle_lstm_trading_strategy.py`
4. Paste into notebook cell and run

### 📚 Method 2: Use Notebook Template (Recommended)
1. Open `kaggle_notebook_template.py`
2. Copy each cell (marked as CELL 1, CELL 2, etc.)
3. Create separate Kaggle notebook cells for each
4. Run cells in order

### 📂 Method 3: Upload as Dataset
1. Upload `kaggle_lstm_trading_strategy.py` as a Kaggle dataset
2. In a new notebook:
```python
!cp /kaggle/input/your-dataset/kaggle_lstm_trading_strategy.py .
exec(open('kaggle_lstm_trading_strategy.py').read())
```

## 📊 Using Your Own Data

Replace the sample data section with:
```python
# Load your CSV file
df = pd.read_csv('/kaggle/input/your-dataset/your-file.csv')

# Ensure proper column names: Date, Open, High, Low, Close, Volume
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)
```

## ⚡ Expected Runtime
- Small dataset (1-2 years): 5-10 minutes
- Medium dataset (3-5 years): 10-20 minutes  
- Large dataset (5+ years): 20-30 minutes

## 🎯 Output You'll Get
1. **Model Metrics**: MAE, RMSE, R², Directional Accuracy
2. **Trading Signals**: BUY/SELL with confidence scores
3. **Visualizations**: 4 comprehensive plots
4. **Performance Analysis**: Signal accuracy, profitability stats

## 🔧 Common Issues & Solutions

**"Column not found"**: Check your CSV has: Date, Open, High, Low, Close, Volume
**"Insufficient data"**: Need minimum 200+ rows for good results
**"Memory error"**: Reduce sequence_length or batch_size
**"Low accuracy"**: Try different sequence_length (30-90) or target_horizon (3-10)

## 🏆 Pro Tips
- Add markdown cells between code for better presentation
- Comment your analysis for higher notebook scores
- Make notebook public to share with community
- Save important plots and CSV results

Ready to trade with AI! 🚀📈
