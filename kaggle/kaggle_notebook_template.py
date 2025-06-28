# 🚀 LSTM Trading Strategy - Kaggle Notebook Template
# Copy each cell below into separate Kaggle notebook cells

# =============================================================================
# CELL 1: Introduction and Setup
# =============================================================================
"""
# 🚀 LSTM Trading Strategy with 13 Technical Indicators

This notebook implements a complete LSTM trading strategy using 13 specific technical indicators:

📈 **Momentum**: RSI, ROC, Stochastic Oscillator, TSI  
📊 **Volume**: OBV, MFI, PVT  
📉 **Trend**: TEMA, MACD, KAMA  
🌪️ **Volatility**: ATR, Bollinger Bands, Ulcer Index  

The strategy uses deep learning to predict price movements and generate trading signals.
"""

# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from datetime import datetime
import os

# Machine Learning imports
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Deep Learning imports
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Configure environment
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')
plt.style.use('default')
sns.set_palette("husl")

print("🚀 KAGGLE LSTM TRADING STRATEGY")
print("=" * 50)
print(f"TensorFlow version: {tf.__version__}")
print(f"GPU Available: {len(tf.config.list_physical_devices('GPU'))} devices")
print("=" * 50)

# =============================================================================
# CELL 2: Technical Indicators Calculator
# =============================================================================

class TechnicalIndicators:
    """Self-contained technical indicators calculator"""
    
    def __init__(self, df):
        self.df = df.copy()
        
    def calculate_rsi(self, period=14):
        """Relative Strength Index"""
        delta = self.df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return {'RSI': rsi}
    
    def calculate_roc(self, period=12):
        """Rate of Change"""
        roc = ((self.df['Close'] - self.df['Close'].shift(period)) / self.df['Close'].shift(period)) * 100
        return {'ROC': roc}
    
    def calculate_stochastic(self, k_period=14, d_period=3):
        """Stochastic Oscillator"""
        low_min = self.df['Low'].rolling(window=k_period).min()
        high_max = self.df['High'].rolling(window=k_period).max()
        k_percent = 100 * ((self.df['Close'] - low_min) / (high_max - low_min))
        d_percent = k_percent.rolling(window=d_period).mean()
        return {'Stoch_K': k_percent, 'Stoch_D': d_percent}
    
    def calculate_tsi(self, long_period=25, short_period=13):
        """True Strength Index"""
        price_change = self.df['Close'].diff()
        double_smoothed_pc = price_change.ewm(span=long_period).mean().ewm(span=short_period).mean()
        double_smoothed_abs_pc = price_change.abs().ewm(span=long_period).mean().ewm(span=short_period).mean()
        tsi = 100 * (double_smoothed_pc / double_smoothed_abs_pc)
        return {'TSI': tsi}
    
    def calculate_obv(self):
        """On Balance Volume"""
        obv = np.where(self.df['Close'] > self.df['Close'].shift(1), 
                       self.df['Volume'], 
                       np.where(self.df['Close'] < self.df['Close'].shift(1), 
                               -self.df['Volume'], 0)).cumsum()
        return {'OBV': pd.Series(obv, index=self.df.index)}
    
    def calculate_mfi(self, period=14):
        """Money Flow Index"""
        typical_price = (self.df['High'] + self.df['Low'] + self.df['Close']) / 3
        money_flow = typical_price * self.df['Volume']
        
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)
        
        positive_mf = positive_flow.rolling(window=period).sum()
        negative_mf = negative_flow.rolling(window=period).sum()
        
        mfi = 100 - (100 / (1 + (positive_mf / negative_mf)))
        return {'MFI': mfi}
    
    def calculate_pvt(self):
        """Price Volume Trend"""
        pvt = (((self.df['Close'] - self.df['Close'].shift(1)) / self.df['Close'].shift(1)) * self.df['Volume']).cumsum()
        return {'PVT': pvt}
    
    def calculate_tema(self, period=12):
        """Triple Exponential Moving Average"""
        ema1 = self.df['Close'].ewm(span=period).mean()
        ema2 = ema1.ewm(span=period).mean()
        ema3 = ema2.ewm(span=period).mean()
        tema = 3 * ema1 - 3 * ema2 + ema3
        return {'TEMA': tema}
    
    def calculate_macd(self, fast=12, slow=26, signal=9):
        """MACD"""
        ema_fast = self.df['Close'].ewm(span=fast).mean()
        ema_slow = self.df['Close'].ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        return {'MACD': macd}
    
    def calculate_kama(self, period=10):
        """Kaufman's Adaptive Moving Average"""
        change = self.df['Close'].diff(period).abs()
        volatility = self.df['Close'].diff().abs().rolling(window=period).sum()
        er = change / volatility
        
        sc = (er * (2.0 / (2 + 1) - 2.0 / (30 + 1)) + 2.0 / (30 + 1)) ** 2.0
        
        kama = np.zeros(len(self.df))
        kama[period-1] = self.df['Close'].iloc[period-1]
        
        for i in range(period, len(kama)):
            kama[i] = kama[i-1] + sc.iloc[i] * (self.df['Close'].iloc[i] - kama[i-1])
        
        return {'KAMA': pd.Series(kama, index=self.df.index)}
    
    def calculate_atr(self, period=14):
        """Average True Range"""
        high_low = self.df['High'] - self.df['Low']
        high_close = np.abs(self.df['High'] - self.df['Close'].shift())
        low_close = np.abs(self.df['Low'] - self.df['Close'].shift())
        
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = true_range.rolling(window=period).mean()
        return {'ATR': atr}
    
    def calculate_bollinger_bands(self, period=20, std_dev=2):
        """Bollinger Bands"""
        sma = self.df['Close'].rolling(window=period).mean()
        std = self.df['Close'].rolling(window=period).std()
        
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        bb_position = (self.df['Close'] - lower_band) / (upper_band - lower_band)
        
        return {'BB_Position': bb_position}
    
    def calculate_ulcer_index(self, period=14):
        """Ulcer Index"""
        highest_close = self.df['Close'].rolling(window=period).max()
        drawdown = ((self.df['Close'] - highest_close) / highest_close) * 100
        ulcer_index = np.sqrt((drawdown ** 2).rolling(window=period).mean())
        return {'Ulcer_Index': ulcer_index}
    
    def calculate_all_indicators(self):
        """Calculate all 13 indicators"""
        print("🔧 Calculating technical indicators...")
        
        indicators = {}
        
        # Add all indicators
        indicators.update(self.calculate_rsi())
        indicators.update(self.calculate_roc())
        indicators.update(self.calculate_stochastic())
        indicators.update(self.calculate_tsi())
        indicators.update(self.calculate_obv())
        indicators.update(self.calculate_mfi())
        indicators.update(self.calculate_pvt())
        indicators.update(self.calculate_tema())
        indicators.update(self.calculate_macd())
        indicators.update(self.calculate_kama())
        indicators.update(self.calculate_atr())
        indicators.update(self.calculate_bollinger_bands())
        indicators.update(self.calculate_ulcer_index())
        
        # Add to dataframe
        result_df = self.df.copy()
        for name, values in indicators.items():
            result_df[name] = values
        
        print(f"✅ Calculated {len(indicators)} technical indicators")
        return result_df

# =============================================================================
# CELL 3: LSTM Strategy Class
# =============================================================================

class KaggleLSTMStrategy:
    """LSTM Trading Strategy for Kaggle"""
    
    def __init__(self, sequence_length=60, target_horizon=5, test_split=0.2):
        self.sequence_length = sequence_length
        self.target_horizon = target_horizon
        self.test_split = test_split
        self.scaler = MinMaxScaler()
        
        print(f"🤖 LSTM Strategy initialized:")
        print(f"   Sequence Length: {sequence_length} days")
        print(f"   Prediction Horizon: {target_horizon} days")
        print(f"   Test Split: {test_split*100}%")
    
    def prepare_data(self, df):
        """Prepare data for LSTM"""
        print("📋 Preparing data for LSTM...")
        
        # Core features
        features = ['Close', 'RSI', 'ROC', 'Stoch_K', 'Stoch_D', 'TSI',
                   'OBV', 'MFI', 'PVT', 'TEMA', 'MACD', 'KAMA',
                   'ATR', 'BB_Position', 'Ulcer_Index']
        
        # Select available features
        available_features = [f for f in features if f in df.columns]
        print(f"📊 Using {len(available_features)} features: {available_features}")
        
        # Extract and clean data
        data = df[available_features].copy()
        data = data.fillna(method='ffill').fillna(method='bfill')
        data = data.dropna()
        
        # Scale data
        scaled_data = self.scaler.fit_transform(data)
        
        # Create sequences
        X, y = [], []
        for i in range(self.sequence_length, len(scaled_data) - self.target_horizon):
            X.append(scaled_data[i - self.sequence_length:i])
            y.append(scaled_data[i + self.target_horizon, 0])  # Predict Close price
        
        X, y = np.array(X), np.array(y)
        
        # Split data
        split_idx = int(len(X) * (1 - self.test_split))
        self.X_train, self.X_test = X[:split_idx], X[split_idx:]
        self.y_train, self.y_test = y[:split_idx], y[split_idx:]
        
        print(f"✅ Data prepared: Train={len(self.X_train)}, Test={len(self.X_test)}")
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def build_model(self, lstm_units=[100, 50], dropout_rate=0.2):
        """Build LSTM model"""
        print("🏗️ Building LSTM model...")
        
        self.model = Sequential([
            LSTM(lstm_units[0], return_sequences=True, 
                 input_shape=(self.X_train.shape[1], self.X_train.shape[2])),
            Dropout(dropout_rate),
            LSTM(lstm_units[1]),
            Dropout(dropout_rate),
            Dense(50, activation='relu'),
            Dense(1)
        ])
        
        self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        print("✅ Model built successfully")
        return self.model
    
    def train_model(self, epochs=50, batch_size=32):
        """Train the model"""
        print("🎓 Training LSTM model...")
        
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=5)
        ]
        
        self.history = self.model.fit(
            self.X_train, self.y_train,
            epochs=epochs, batch_size=batch_size,
            validation_split=0.2, callbacks=callbacks,
            verbose=1
        )
        
        print("✅ Training completed!")
        return self.history
    
    def make_predictions(self):
        """Make predictions and calculate metrics"""
        print("🔮 Making predictions...")
        
        # Predict
        y_pred_scaled = self.model.predict(self.X_test, verbose=0)
        
        # Inverse transform
        dummy_pred = np.zeros((len(y_pred_scaled), self.scaler.n_features_in_))
        dummy_pred[:, 0] = y_pred_scaled.flatten()
        self.y_pred = self.scaler.inverse_transform(dummy_pred)[:, 0]
        
        dummy_actual = np.zeros((len(self.y_test), self.scaler.n_features_in_))
        dummy_actual[:, 0] = self.y_test
        self.y_actual = self.scaler.inverse_transform(dummy_actual)[:, 0]
        
        # Calculate metrics
        mae = mean_absolute_error(self.y_actual, self.y_pred)
        mse = mean_squared_error(self.y_actual, self.y_pred)
        r2 = r2_score(self.y_actual, self.y_pred)
        
        # Directional accuracy
        actual_direction = np.diff(self.y_actual) > 0
        pred_direction = np.diff(self.y_pred) > 0
        directional_accuracy = np.mean(actual_direction == pred_direction) * 100
        
        self.metrics = {
            'MAE': round(mae, 4),
            'RMSE': round(np.sqrt(mse), 4),
            'R2_Score': round(r2, 4),
            'Directional_Accuracy': round(directional_accuracy, 2)
        }
        
        print("📊 Prediction Metrics:")
        for metric, value in self.metrics.items():
            print(f"   {metric}: {value}")
        
        return self.y_pred, self.y_actual

# =============================================================================
# CELL 4: Sample Data Generator (Use this if you don't have your own data)
# =============================================================================

def create_sample_data():
    """Create realistic sample stock data"""
    print("📊 Creating sample stock data...")
    
    # Generate dates (weekdays only)
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
    dates = dates[dates.weekday < 5]  # Remove weekends
    
    # Generate realistic price data
    np.random.seed(42)
    n_days = len(dates)
    base_price = 100
    
    # Random walk with drift
    daily_returns = np.random.normal(0.0005, 0.02, n_days)
    prices = [base_price]
    
    for i in range(1, n_days):
        new_price = prices[-1] * (1 + daily_returns[i])
        prices.append(max(new_price, 1))  # Ensure positive prices
    
    # Create OHLCV data
    df = pd.DataFrame({
        'Date': dates,
        'Open': [p * (1 + np.random.normal(0, 0.005)) for p in prices],
        'High': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'Low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'Close': prices,
        'Volume': np.random.randint(100000, 1000000, n_days)
    })
    
    # Ensure OHLC consistency
    df['High'] = df[['Open', 'High', 'Close']].max(axis=1)
    df['Low'] = df[['Open', 'Low', 'Close']].min(axis=1)
    
    print(f"✅ Created {len(df)} days of sample data")
    print(f"   Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
    print(f"   Price range: ${df['Close'].min():.2f} to ${df['Close'].max():.2f}")
    
    return df

# =============================================================================
# CELL 5: Load Your Data (Modify this section for your own data)
# =============================================================================

# Option A: Use sample data (for demonstration)
df = create_sample_data()

# Option B: Load your own CSV data (uncomment and modify)
# df = pd.read_csv('/kaggle/input/your-dataset/your-file.csv')
# df['Date'] = pd.to_datetime(df['Date'])
# df = df.sort_values('Date').reset_index(drop=True)

# Display basic info
print("\n📊 Data Overview:")
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print("\nFirst few rows:")
print(df.head())

print("\nBasic statistics:")
print(df[['Open', 'High', 'Low', 'Close', 'Volume']].describe())

# =============================================================================
# CELL 6: Calculate Technical Indicators
# =============================================================================

# Calculate all technical indicators
tech_calc = TechnicalIndicators(df)
df_with_indicators = tech_calc.calculate_all_indicators()

print(f"\n📊 Data with indicators shape: {df_with_indicators.shape}")
print(f"New columns added: {df_with_indicators.shape[1] - df.shape[1]}")

# Show available indicators
indicator_columns = [col for col in df_with_indicators.columns 
                    if col not in ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
print(f"\nTechnical indicators calculated: {len(indicator_columns)}")
for i, col in enumerate(indicator_columns, 1):
    print(f"  {i:2d}. {col}")

# =============================================================================
# CELL 7: Initialize and Train LSTM Strategy
# =============================================================================

# Initialize strategy
strategy = KaggleLSTMStrategy(
    sequence_length=60,    # Use 60 days of history
    target_horizon=5,      # Predict 5 days ahead
    test_split=0.2         # 20% for testing
)

# Prepare data
X_train, X_test, y_train, y_test = strategy.prepare_data(df_with_indicators)

# Build model
model = strategy.build_model(
    lstm_units=[100, 50],  # Two LSTM layers
    dropout_rate=0.2       # 20% dropout
)

# Train model
history = strategy.train_model(
    epochs=50,            # Train for 50 epochs
    batch_size=32         # Batch size of 32
)

# =============================================================================
# CELL 8: Make Predictions and Generate Signals
# =============================================================================

# Make predictions
y_pred, y_actual = strategy.make_predictions()

# Generate trading signals
print("\n🎯 Generating trading signals...")

signals = []
for i in range(min(len(y_pred), len(y_actual)) - strategy.target_horizon):
    current_price = y_actual[i]
    predicted_price = y_pred[i]
    
    if i + strategy.target_horizon < len(y_actual):
        future_price = y_actual[i + strategy.target_horizon]
    else:
        continue
    
    # Calculate changes
    predicted_change = (predicted_price - current_price) / current_price
    actual_change = (future_price - current_price) / current_price
    
    # Generate signal
    signal_type = 'BUY' if predicted_change > 0.01 else 'SELL'
    actual_trend = 'BUY' if actual_change > 0 else 'SELL'
    confidence = min(abs(predicted_change) * 100, 10)
    
    signals.append({
        'Day': i,
        'Current_Price': round(current_price, 2),
        'Predicted_Price': round(predicted_price, 2),
        'Actual_Future_Price': round(future_price, 2),
        'Signal': signal_type,
        'Confidence_%': round(confidence, 2),
        'Predicted_Change_%': round(predicted_change * 100, 2),
        'Actual_Change_%': round(actual_change * 100, 2),
        'Correct': signal_type == actual_trend
    })

signals_df = pd.DataFrame(signals)

# Calculate performance
total_signals = len(signals_df)
correct_signals = signals_df['Correct'].sum()
signal_accuracy = (correct_signals / total_signals) * 100

print(f"✅ Generated {total_signals} trading signals")
print(f"📊 Signal Accuracy: {signal_accuracy:.2f}%")

# =============================================================================
# CELL 9: Results and Visualizations
# =============================================================================

# Display results summary
print("\n" + "="*60)
print("🎉 FINAL RESULTS SUMMARY")
print("="*60)

print("\n📊 Model Performance:")
for metric, value in strategy.metrics.items():
    print(f"  {metric}: {value}")

print(f"\n🎯 Trading Performance:")
print(f"  Total Signals: {total_signals}")
print(f"  Correct Predictions: {correct_signals}")
print(f"  Signal Accuracy: {signal_accuracy:.2f}%")

# Signal type breakdown
buy_signals = signals_df[signals_df['Signal'] == 'BUY']
sell_signals = signals_df[signals_df['Signal'] == 'SELL']

buy_accuracy = (buy_signals['Correct'].sum() / len(buy_signals) * 100) if len(buy_signals) > 0 else 0
sell_accuracy = (sell_signals['Correct'].sum() / len(sell_signals) * 100) if len(sell_signals) > 0 else 0

print(f"\n📈 Signal Breakdown:")
print(f"  BUY Signals: {len(buy_signals)} (Accuracy: {buy_accuracy:.1f}%)")
print(f"  SELL Signals: {len(sell_signals)} (Accuracy: {sell_accuracy:.1f}%)")

# Show sample signals
print(f"\n📋 Sample Trading Signals:")
display_cols = ['Day', 'Signal', 'Confidence_%', 'Predicted_Change_%', 
               'Actual_Change_%', 'Correct']
print(signals_df[display_cols].head(10).to_string(index=False))

# =============================================================================
# CELL 10: Create Visualizations
# =============================================================================

# Create comprehensive plots
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('LSTM Trading Strategy Results', fontsize=16, fontweight='bold')

# 1. Predictions vs Actual
sample_size = min(100, len(y_actual))
axes[0, 0].plot(y_actual[:sample_size], label='Actual', alpha=0.8, linewidth=2)
axes[0, 0].plot(y_pred[:sample_size], label='Predicted', alpha=0.8, linewidth=2)
axes[0, 0].set_title('Price Predictions vs Actual (First 100 days)')
axes[0, 0].set_xlabel('Days')
axes[0, 0].set_ylabel('Price ($)')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. Training History
axes[0, 1].plot(history.history['loss'], label='Training Loss', linewidth=2)
if 'val_loss' in history.history:
    axes[0, 1].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
axes[0, 1].set_title('Model Training History')
axes[0, 1].set_xlabel('Epochs')
axes[0, 1].set_ylabel('Loss')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 3. Signal Accuracy by Type
signal_accuracy_by_type = signals_df.groupby('Signal')['Correct'].mean() * 100
colors = ['red' if signal == 'SELL' else 'green' for signal in signal_accuracy_by_type.index]
axes[1, 0].bar(signal_accuracy_by_type.index, signal_accuracy_by_type.values, color=colors, alpha=0.7)
axes[1, 0].set_title('Signal Accuracy by Type')
axes[1, 0].set_ylabel('Accuracy (%)')
axes[1, 0].grid(True, alpha=0.3)

# 4. Return Distribution
axes[1, 1].hist(signals_df['Actual_Change_%'], bins=20, alpha=0.7, color='blue', edgecolor='black')
axes[1, 1].axvline(0, color='red', linestyle='--', linewidth=2, label='Break-even')
axes[1, 1].axvline(signals_df['Actual_Change_%'].mean(), color='green', 
                  linestyle='--', linewidth=2, label=f'Mean: {signals_df["Actual_Change_%"].mean():.2f}%')
axes[1, 1].set_title('Distribution of Actual Returns')
axes[1, 1].set_xlabel('Return (%)')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# =============================================================================
# CELL 11: Additional Analysis and Export
# =============================================================================

# Performance by confidence level
print("\n📊 Performance by Confidence Level:")
confidence_bins = pd.cut(signals_df['Confidence_%'], bins=3, labels=['Low', 'Medium', 'High'])
conf_performance = signals_df.groupby(confidence_bins)['Correct'].agg(['count', 'sum', 'mean'])
conf_performance['accuracy_%'] = conf_performance['mean'] * 100
print(conf_performance[['count', 'sum', 'accuracy_%']])

# Monthly performance analysis
if 'Date' in df_with_indicators.columns:
    # Add dates to signals for temporal analysis
    test_start_idx = len(df_with_indicators) - len(y_actual)
    signal_dates = df_with_indicators['Date'].iloc[test_start_idx:test_start_idx + len(signals_df)]
    signals_df['Date'] = signal_dates.reset_index(drop=True)
    signals_df['Month'] = pd.to_datetime(signals_df['Date']).dt.to_period('M')
    
    monthly_performance = signals_df.groupby('Month')['Correct'].agg(['count', 'mean'])
    monthly_performance['accuracy_%'] = monthly_performance['mean'] * 100
    print(f"\n📅 Monthly Performance (last 12 months):")
    print(monthly_performance.tail(12)[['count', 'accuracy_%']])

# Summary statistics
print(f"\n📈 Return Statistics:")
print(f"  Average Predicted Return: {signals_df['Predicted_Change_%'].mean():.2f}%")
print(f"  Average Actual Return: {signals_df['Actual_Change_%'].mean():.2f}%")
print(f"  Best Trade: {signals_df['Actual_Change_%'].max():.2f}%")
print(f"  Worst Trade: {signals_df['Actual_Change_%'].min():.2f}%")
print(f"  Return Volatility: {signals_df['Actual_Change_%'].std():.2f}%")

# Profitable trades analysis
profitable_trades = signals_df[signals_df['Actual_Change_%'] > 0]
losing_trades = signals_df[signals_df['Actual_Change_%'] < 0]

print(f"\n💰 Profitability Analysis:")
print(f"  Profitable Trades: {len(profitable_trades)} ({len(profitable_trades)/len(signals_df)*100:.1f}%)")
print(f"  Losing Trades: {len(losing_trades)} ({len(losing_trades)/len(signals_df)*100:.1f}%)")
if len(profitable_trades) > 0:
    print(f"  Average Profit: {profitable_trades['Actual_Change_%'].mean():.2f}%")
if len(losing_trades) > 0:
    print(f"  Average Loss: {losing_trades['Actual_Change_%'].mean():.2f}%")

print(f"\n✅ Analysis Complete! 🚀")
print(f"📁 You can save the results using:")
print(f"   signals_df.to_csv('lstm_trading_signals.csv', index=False)")

# Uncomment to save results
# signals_df.to_csv('lstm_trading_signals.csv', index=False)
# print("💾 Results saved to lstm_trading_signals.csv")
