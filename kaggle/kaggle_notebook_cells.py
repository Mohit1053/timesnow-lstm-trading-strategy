# KAGGLE NOTEBOOK: Focused LSTM Trading Strategy
# 
# This notebook implements a complete LSTM trading strategy using 13 specific technical indicators.
# Simply upload your CSV data and run all cells to get complete trading analysis.

# =============================================================================
# CELL 1: Install Required Packages
# =============================================================================

import subprocess
import sys

def install_packages():
    """Install required packages for the LSTM trading strategy."""
    packages = [
        'tensorflow==2.13.0',
        'scikit-learn',
        'TA-Lib',
        'matplotlib',
        'seaborn',
        'numpy',
        'pandas'
    ]
    
    for package in packages:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package, '--quiet'])
    
    print("✅ All packages installed successfully!")

# Uncomment the line below if running for the first time
# install_packages()

# =============================================================================
# CELL 2: Import Libraries and Configure Settings
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import talib

# Configure plot style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("📊 KAGGLE FOCUSED LSTM TRADING STRATEGY")
print("=" * 50)
print("🚀 All libraries imported successfully!")

# =============================================================================
# CELL 3: Configuration Settings
# =============================================================================

# 🔧 CONFIGURATION - UPDATE THESE FOR YOUR DATA
CONFIG = {
    # Data settings
    'data_path': '/kaggle/input/your-dataset/priceData5Year.csv',  # UPDATE THIS
    'date_column': 'Date',
    'output_dir': '/kaggle/working',
    
    # LSTM model settings
    'sequence_length': 60,
    'lstm_units': 50,
    'dropout_rate': 0.2,
    'epochs': 30,  # Reduced for faster execution
    'batch_size': 32,
    'validation_split': 0.2,
    'learning_rate': 0.001,
    
    # Trading settings
    'initial_capital': 100000,
    'transaction_cost': 0.001,
    'max_position_size': 0.95,
    'stop_loss': 0.05,
    'take_profit': 0.15
}

# The 13 focused technical indicators
INDICATORS = {
    'momentum': ['RSI', 'ROC', 'STOCH_K', 'STOCH_D', 'TSI'],
    'volume': ['OBV', 'MFI', 'PVT'],
    'trend': ['TEMA', 'MACD', 'MACD_Signal', 'MACD_Hist', 'KAMA'],
    'volatility': ['ATR', 'BB_Width', 'BB_Position', 'ULCER']
}

print("⚙️ Configuration loaded:")
print(f"   📁 Data path: {CONFIG['data_path']}")
print(f"   🧠 LSTM units: {CONFIG['lstm_units']}")
print(f"   💰 Initial capital: ${CONFIG['initial_capital']:,}")
print(f"   📊 Using {sum(len(v) for v in INDICATORS.values())} indicators")

# =============================================================================
# CELL 4: Data Loading and Preprocessing
# =============================================================================

def load_and_prepare_data(file_path):
    """Load and prepare the price data."""
    print("📂 Loading data...")
    
    # Load data
    df = pd.read_csv(file_path)
    
    # Handle date column
    if CONFIG['date_column'] in df.columns:
        df[CONFIG['date_column']] = pd.to_datetime(df[CONFIG['date_column']])
        df.set_index(CONFIG['date_column'], inplace=True)
    else:
        df.index = pd.to_datetime(df.index)
    
    # Ensure required columns exist
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")
    
    print(f"   ✅ Data loaded: {df.shape}")
    print(f"   📅 Date range: {df.index.min()} to {df.index.max()}")
    
    return df

# Load data (update the path in CONFIG above)
try:
    df = load_and_prepare_data(CONFIG['data_path'])
    print("\n📊 Data Overview:")
    print(df.head())
    print(f"\n📈 Price Statistics:")
    print(df['Close'].describe())
except Exception as e:
    print(f"❌ Error loading data: {e}")
    print("💡 Please update the 'data_path' in CONFIG section above")

# =============================================================================
# CELL 5: Technical Indicators Calculation
# =============================================================================

def calculate_all_indicators(df):
    """Calculate all 13 focused technical indicators."""
    data = df.copy()
    
    print("🔧 Calculating technical indicators...")
    
    # Convert to numpy arrays
    open_prices = data['Open'].astype(float).values
    high_prices = data['High'].astype(float).values
    low_prices = data['Low'].astype(float).values
    close_prices = data['Close'].astype(float).values
    volume = data['Volume'].astype(float).values
    
    # Momentum Indicators
    print("   📈 Momentum indicators...")
    data['RSI'] = talib.RSI(close_prices, timeperiod=14)
    data['ROC'] = talib.ROC(close_prices, timeperiod=10)
    data['STOCH_K'], data['STOCH_D'] = talib.STOCH(high_prices, low_prices, close_prices)
    data['TSI'] = talib.TRIX(close_prices, timeperiod=14)
    
    # Volume Indicators
    print("   📊 Volume indicators...")
    data['OBV'] = talib.OBV(close_prices, volume)
    data['MFI'] = talib.MFI(high_prices, low_prices, close_prices, volume)
    data['PVT'] = ((close_prices - np.roll(close_prices, 1)) / np.roll(close_prices, 1) * volume).cumsum()
    
    # Trend Indicators
    print("   📉 Trend indicators...")
    data['TEMA'] = talib.TEMA(close_prices, timeperiod=14)
    data['MACD'], data['MACD_Signal'], data['MACD_Hist'] = talib.MACD(close_prices)
    data['KAMA'] = talib.KAMA(close_prices, timeperiod=30)
    
    # Volatility Indicators
    print("   🌪️ Volatility indicators...")
    data['ATR'] = talib.ATR(high_prices, low_prices, close_prices)
    data['BB_Upper'], data['BB_Middle'], data['BB_Lower'] = talib.BBANDS(close_prices)
    data['BB_Width'] = data['BB_Upper'] - data['BB_Lower']
    data['BB_Position'] = (close_prices - data['BB_Lower']) / (data['BB_Upper'] - data['BB_Lower'])
    
    # Ulcer Index (manual calculation)
    max_prices = pd.Series(close_prices).rolling(window=14).max()
    drawdown = ((close_prices - max_prices) / max_prices) * 100
    data['ULCER'] = np.sqrt((drawdown ** 2).rolling(window=14).mean()).values
    
    print("   ✅ All indicators calculated successfully!")
    
    return data

# Calculate indicators
df_indicators = calculate_all_indicators(df)

# Show indicator statistics
feature_cols = []
for category in INDICATORS.values():
    feature_cols.extend(category)

print(f"\n📊 Indicators Overview:")
print(df_indicators[feature_cols].describe())

# =============================================================================
# CELL 6: LSTM Model Implementation
# =============================================================================

class SimpleLSTMStrategy:
    """Simplified LSTM Strategy for Kaggle."""
    
    def __init__(self, feature_columns):
        self.feature_columns = feature_columns
        self.scaler = MinMaxScaler()
        self.model = None
        
    def prepare_sequences(self, data, sequence_length):
        """Prepare sequences for LSTM training."""
        print(f"🔄 Preparing sequences (length: {sequence_length})...")
        
        # Select features and target
        features = data[self.feature_columns].values
        target = data['Close'].values
        
        # Scale data
        scaled_features = self.scaler.fit_transform(features)
        
        # Create sequences
        X, y = [], []
        for i in range(sequence_length, len(scaled_features)):
            X.append(scaled_features[i-sequence_length:i])
            y.append(target[i])
        
        return np.array(X), np.array(y)
    
    def build_model(self, input_shape):
        """Build LSTM model."""
        model = Sequential([
            LSTM(CONFIG['lstm_units'], return_sequences=True, input_shape=input_shape),
            Dropout(CONFIG['dropout_rate']),
            LSTM(CONFIG['lstm_units']//2, return_sequences=False),
            Dropout(CONFIG['dropout_rate']),
            Dense(25),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=CONFIG['learning_rate']), 
                     loss='mse')
        return model
    
    def train(self, X, y):
        """Train the model."""
        print("🎯 Training LSTM model...")
        
        self.model = self.build_model((X.shape[1], X.shape[2]))
        
        history = self.model.fit(
            X, y,
            epochs=CONFIG['epochs'],
            batch_size=CONFIG['batch_size'],
            validation_split=CONFIG['validation_split'],
            verbose=1
        )
        
        return history
    
    def predict(self, X):
        """Make predictions."""
        return self.model.predict(X).flatten()

# Initialize strategy
feature_columns = []
for category in INDICATORS.values():
    feature_columns.extend(category)

strategy = SimpleLSTMStrategy(feature_columns)

# Prepare data
clean_data = df_indicators.dropna()
X, y = strategy.prepare_sequences(clean_data, CONFIG['sequence_length'])

print(f"📊 Training data shape: X={X.shape}, y={y.shape}")

# =============================================================================
# CELL 7: Model Training and Prediction
# =============================================================================

# Train model
history = strategy.train(X, y)

# Make predictions
print("🔮 Making predictions...")
predictions = strategy.predict(X)

# Calculate metrics
y_actual = y
mse = mean_squared_error(y_actual, predictions)
mae = mean_absolute_error(y_actual, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_actual, predictions)

print(f"\n📊 Model Performance:")
print(f"   MSE: {mse:.2f}")
print(f"   MAE: {mae:.2f}")
print(f"   RMSE: {rmse:.2f}")
print(f"   R²: {r2:.4f}")

# Plot training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(y_actual, predictions, alpha=0.5)
plt.plot([y_actual.min(), y_actual.max()], [y_actual.min(), y_actual.max()], 'r--')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Predictions vs Actual')

plt.tight_layout()
plt.show()

# =============================================================================
# CELL 8: Trading Simulation
# =============================================================================

def simulate_trading_strategy(data, predictions, start_idx):
    """Simulate trading based on LSTM predictions."""
    print("💼 Simulating trading strategy...")
    
    # Prepare results dataframe
    results = data.iloc[start_idx:start_idx+len(predictions)].copy()
    results['Prediction'] = predictions
    results['Signal'] = 0
    results['Position'] = 0
    results['Portfolio_Value'] = CONFIG['initial_capital']
    results['Returns'] = 0.0
    
    # Generate signals
    for i in range(1, len(results)):
        if results.iloc[i]['Prediction'] > results.iloc[i-1]['Close']:
            results.iloc[i, results.columns.get_loc('Signal')] = 1  # Buy
        elif results.iloc[i]['Prediction'] < results.iloc[i-1]['Close']:
            results.iloc[i, results.columns.get_loc('Signal')] = -1  # Sell
    
    # Simulate trading
    capital = CONFIG['initial_capital']
    position = 0
    shares = 0
    trades = []
    
    for i in range(1, len(results)):
        current_price = results.iloc[i]['Close']
        signal = results.iloc[i]['Signal']
        
        if signal == 1 and position <= 0 and capital > 0:  # Buy
            shares_to_buy = int((capital * CONFIG['max_position_size']) // current_price)
            if shares_to_buy > 0:
                cost = shares_to_buy * current_price * (1 + CONFIG['transaction_cost'])
                if cost <= capital:
                    capital -= cost
                    shares += shares_to_buy
                    position = 1
                    trades.append({
                        'Date': results.index[i],
                        'Action': 'BUY',
                        'Price': current_price,
                        'Shares': shares_to_buy,
                        'Cost': cost
                    })
        
        elif signal == -1 and position >= 0 and shares > 0:  # Sell
            revenue = shares * current_price * (1 - CONFIG['transaction_cost'])
            capital += revenue
            trades.append({
                'Date': results.index[i],
                'Action': 'SELL',
                'Price': current_price,
                'Shares': shares,
                'Revenue': revenue
            })
            shares = 0
            position = -1
        
        # Update portfolio value
        portfolio_value = capital + (shares * current_price)
        results.iloc[i, results.columns.get_loc('Portfolio_Value')] = portfolio_value
        results.iloc[i, results.columns.get_loc('Position')] = position
        
        # Calculate returns
        if i > 0:
            prev_value = results.iloc[i-1]['Portfolio_Value']
            if prev_value > 0:
                daily_return = (portfolio_value - prev_value) / prev_value
                results.iloc[i, results.columns.get_loc('Returns')] = daily_return
    
    return results, pd.DataFrame(trades)

# Run simulation
start_idx = CONFIG['sequence_length']
results, trading_log = simulate_trading_strategy(clean_data, predictions, start_idx)

print(f"📊 Trading Summary:")
print(f"   Total trades: {len(trading_log)}")
print(f"   Final portfolio value: ${results['Portfolio_Value'].iloc[-1]:,.2f}")

total_return = (results['Portfolio_Value'].iloc[-1] - CONFIG['initial_capital']) / CONFIG['initial_capital']
print(f"   Total return: {total_return:.2%}")

# =============================================================================
# CELL 9: Performance Analysis and Visualization
# =============================================================================

def analyze_performance(results, trading_log):
    """Analyze trading performance and create visualizations."""
    
    # Calculate metrics
    final_value = results['Portfolio_Value'].iloc[-1]
    total_return = (final_value - CONFIG['initial_capital']) / CONFIG['initial_capital']
    
    returns = results['Returns'].dropna()
    if len(returns) > 0:
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        max_drawdown = ((results['Portfolio_Value'] / results['Portfolio_Value'].cummax()) - 1).min()
    else:
        sharpe_ratio = 0
        max_drawdown = 0
    
    metrics = {
        'Total Return': f"{total_return:.2%}",
        'Final Value': f"${final_value:,.2f}",
        'Profit/Loss': f"${final_value - CONFIG['initial_capital']:,.2f}",
        'Sharpe Ratio': f"{sharpe_ratio:.3f}",
        'Max Drawdown': f"{max_drawdown:.2%}",
        'Total Trades': len(trading_log)
    }
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Price vs Predictions
    axes[0, 0].plot(results.index, results['Close'], label='Actual Price', alpha=0.8)
    axes[0, 0].plot(results.index, results['Prediction'], label='LSTM Prediction', alpha=0.8)
    axes[0, 0].set_title('Actual vs Predicted Prices')
    axes[0, 0].legend()
    
    # Portfolio Value
    axes[0, 1].plot(results.index, results['Portfolio_Value'], color='green')
    axes[0, 1].set_title('Portfolio Value Over Time')
    axes[0, 1].set_ylabel('Portfolio Value ($)')
    
    # Trading Signals
    buy_signals = results[results['Signal'] == 1]
    sell_signals = results[results['Signal'] == -1]
    
    axes[1, 0].plot(results.index, results['Close'], alpha=0.7)
    axes[1, 0].scatter(buy_signals.index, buy_signals['Close'], 
                       color='green', marker='^', label='Buy', s=30)
    axes[1, 0].scatter(sell_signals.index, sell_signals['Close'], 
                       color='red', marker='v', label='Sell', s=30)
    axes[1, 0].set_title('Trading Signals')
    axes[1, 0].legend()
    
    # Returns Distribution
    if len(returns) > 0:
        axes[1, 1].hist(returns, bins=30, alpha=0.7, color='blue')
        axes[1, 1].set_title('Daily Returns Distribution')
        axes[1, 1].set_xlabel('Daily Returns')
    
    plt.tight_layout()
    plt.show()
    
    return metrics

# Analyze performance
performance_metrics = analyze_performance(results, trading_log)

print("\n💰 FINAL PERFORMANCE SUMMARY:")
print("=" * 40)
for metric, value in performance_metrics.items():
    print(f"{metric}: {value}")

# =============================================================================
# CELL 10: Save Results
# =============================================================================

# Save results to CSV files
results_file = f"{CONFIG['output_dir']}/kaggle_lstm_results.csv"
results.to_csv(results_file)

trading_file = f"{CONFIG['output_dir']}/kaggle_trading_log.csv"
trading_log.to_csv(trading_file, index=False)

# Save summary report
summary_file = f"{CONFIG['output_dir']}/kaggle_performance_summary.txt"
with open(summary_file, 'w') as f:
    f.write("KAGGLE LSTM TRADING STRATEGY - PERFORMANCE REPORT\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Data Period: {clean_data.index.min()} to {clean_data.index.max()}\n\n")
    
    f.write("PERFORMANCE METRICS:\n")
    f.write("-" * 30 + "\n")
    for metric, value in performance_metrics.items():
        f.write(f"{metric}: {value}\n")
    
    f.write(f"\nMODEL METRICS:\n")
    f.write("-" * 30 + "\n")
    f.write(f"MSE: {mse:.2f}\n")
    f.write(f"RMSE: {rmse:.2f}\n")
    f.write(f"MAE: {mae:.2f}\n")
    f.write(f"R²: {r2:.4f}\n")
    
    f.write(f"\nCONFIGURATION:\n")
    f.write("-" * 30 + "\n")
    for key, value in CONFIG.items():
        f.write(f"{key}: {value}\n")

print(f"\n💾 Results saved:")
print(f"   📊 Results: {results_file}")
print(f"   📋 Trading log: {trading_file}")
print(f"   📄 Summary: {summary_file}")

print(f"\n🎉 ANALYSIS COMPLETE!")
print("=" * 50)
print("✅ All files generated successfully!")
print("📈 Check the visualizations above for insights")
print("💡 Modify CONFIG settings and re-run for different strategies")
