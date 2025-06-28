#!/usr/bin/env python3
"""
LSTM Trading Strategy for Kaggle Environment

This script is optimized to run on Kaggle with the 13 specific technical indicators:

📈 Momentum Indicators:
- RSI (Relative Strength Index)
- ROC (Rate of Change)
- Stochastic Oscillator
- TSI (True Strength Index)

📊 Volume Indicators:
- OBV (On Balance Volume)
- MFI (Money Flow Index)
- PVT (Price Volume Trend)

📉 Trend Indicators:
- TEMA (Triple Exponential Moving Average)
- MACD (Moving Average Convergence Divergence)
- KAMA (Kaufman's Adaptive Moving Average)

🌪️ Volatility Indicators:
- ATR (Average True Range)
- Bollinger Bands
- Ulcer Index

Kaggle-specific optimizations:
- Self-contained code (no external imports from local files)
- Efficient memory usage
- GPU acceleration support
- Built-in data validation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from datetime import datetime
import os

# Machine Learning imports
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Deep Learning imports
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

print("🚀 KAGGLE LSTM TRADING STRATEGY")
print("=" * 50)
print(f"TensorFlow version: {tf.__version__}")
print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")
print("=" * 50)

class TechnicalIndicators:
    """Self-contained technical indicators calculator optimized for Kaggle"""
    
    def __init__(self, df):
        self.df = df.copy()
        
    def calculate_rsi(self, period=14):
        """Calculate Relative Strength Index"""
        delta = self.df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return {'RSI': rsi}
    
    def calculate_roc(self, period=12):
        """Calculate Rate of Change"""
        roc = ((self.df['Close'] - self.df['Close'].shift(period)) / self.df['Close'].shift(period)) * 100
        return {'ROC': roc}
    
    def calculate_stochastic(self, k_period=14, d_period=3):
        """Calculate Stochastic Oscillator"""
        low_min = self.df['Low'].rolling(window=k_period).min()
        high_max = self.df['High'].rolling(window=k_period).max()
        k_percent = 100 * ((self.df['Close'] - low_min) / (high_max - low_min))
        d_percent = k_percent.rolling(window=d_period).mean()
        return {'Stoch_K': k_percent, 'Stoch_D': d_percent}
    
    def calculate_tsi(self, long_period=25, short_period=13):
        """Calculate True Strength Index"""
        price_change = self.df['Close'].diff()
        double_smoothed_pc = price_change.ewm(span=long_period).mean().ewm(span=short_period).mean()
        double_smoothed_abs_pc = price_change.abs().ewm(span=long_period).mean().ewm(span=short_period).mean()
        tsi = 100 * (double_smoothed_pc / double_smoothed_abs_pc)
        return {'TSI': tsi}
    
    def calculate_obv(self):
        """Calculate On Balance Volume"""
        obv = np.where(self.df['Close'] > self.df['Close'].shift(1), 
                       self.df['Volume'], 
                       np.where(self.df['Close'] < self.df['Close'].shift(1), 
                               -self.df['Volume'], 0)).cumsum()
        return {'OBV': pd.Series(obv, index=self.df.index)}
    
    def calculate_mfi(self, period=14):
        """Calculate Money Flow Index"""
        typical_price = (self.df['High'] + self.df['Low'] + self.df['Close']) / 3
        money_flow = typical_price * self.df['Volume']
        
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)
        
        positive_mf = positive_flow.rolling(window=period).sum()
        negative_mf = negative_flow.rolling(window=period).sum()
        
        mfi = 100 - (100 / (1 + (positive_mf / negative_mf)))
        return {'MFI': mfi}
    
    def calculate_pvt(self):
        """Calculate Price Volume Trend"""
        pvt = (((self.df['Close'] - self.df['Close'].shift(1)) / self.df['Close'].shift(1)) * self.df['Volume']).cumsum()
        return {'PVT': pvt}
    
    def calculate_tema(self, period=12):
        """Calculate Triple Exponential Moving Average"""
        ema1 = self.df['Close'].ewm(span=period).mean()
        ema2 = ema1.ewm(span=period).mean()
        ema3 = ema2.ewm(span=period).mean()
        tema = 3 * ema1 - 3 * ema2 + ema3
        return {'TEMA': tema}
    
    def calculate_macd(self, fast=12, slow=26, signal=9):
        """Calculate MACD"""
        ema_fast = self.df['Close'].ewm(span=fast).mean()
        ema_slow = self.df['Close'].ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal).mean()
        histogram = macd - signal_line
        return {'MACD': macd, 'MACD_Signal': signal_line, 'MACD_Histogram': histogram}
    
    def calculate_kama(self, period=10):
        """Calculate Kaufman's Adaptive Moving Average"""
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
        """Calculate Average True Range"""
        high_low = self.df['High'] - self.df['Low']
        high_close = np.abs(self.df['High'] - self.df['Close'].shift())
        low_close = np.abs(self.df['Low'] - self.df['Close'].shift())
        
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = true_range.rolling(window=period).mean()
        return {'ATR': atr}
    
    def calculate_bollinger_bands(self, period=20, std_dev=2):
        """Calculate Bollinger Bands"""
        sma = self.df['Close'].rolling(window=period).mean()
        std = self.df['Close'].rolling(window=period).std()
        
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        bb_position = (self.df['Close'] - lower_band) / (upper_band - lower_band)
        
        return {
            'BB_Upper': upper_band,
            'BB_Lower': lower_band,
            'BB_Middle': sma,
            'BB_Position': bb_position
        }
    
    def calculate_ulcer_index(self, period=14):
        """Calculate Ulcer Index"""
        highest_close = self.df['Close'].rolling(window=period).max()
        drawdown = ((self.df['Close'] - highest_close) / highest_close) * 100
        ulcer_index = np.sqrt((drawdown ** 2).rolling(window=period).mean())
        return {'Ulcer_Index': ulcer_index}
    
    def calculate_all_indicators(self):
        """Calculate all 13 specified indicators"""
        print("🔧 Calculating technical indicators...")
        
        indicators = {}
        
        # Momentum Indicators
        print("  📈 Momentum indicators...")
        indicators.update(self.calculate_rsi())
        indicators.update(self.calculate_roc())
        indicators.update(self.calculate_stochastic())
        indicators.update(self.calculate_tsi())
        
        # Volume Indicators
        print("  📊 Volume indicators...")
        indicators.update(self.calculate_obv())
        indicators.update(self.calculate_mfi())
        indicators.update(self.calculate_pvt())
        
        # Trend Indicators
        print("  📉 Trend indicators...")
        indicators.update(self.calculate_tema())
        indicators.update(self.calculate_macd())
        indicators.update(self.calculate_kama())
        
        # Volatility Indicators
        print("  🌪️ Volatility indicators...")
        indicators.update(self.calculate_atr())
        indicators.update(self.calculate_bollinger_bands())
        indicators.update(self.calculate_ulcer_index())
        
        # Add to dataframe
        result_df = self.df.copy()
        for name, values in indicators.items():
            result_df[name] = values
        
        print(f"✅ Calculated {len(indicators)} technical indicators")
        return result_df

class KaggleLSTMTradingStrategy:
    """LSTM Trading Strategy optimized for Kaggle environment"""
    
    def __init__(self, sequence_length=60, target_horizon=5, test_split=0.2):
        self.sequence_length = sequence_length
        self.target_horizon = target_horizon
        self.test_split = test_split
        
        # Initialize components
        self.scaler = MinMaxScaler()
        self.model = None
        self.history = None
        
        print(f"🤖 Initialized Kaggle LSTM Strategy")
        print(f"   Sequence Length: {sequence_length}")
        print(f"   Target Horizon: {target_horizon} days")
        print(f"   Test Split: {test_split*100}%")
    
    def prepare_data(self, df):
        """Prepare data for LSTM training"""
        print("📋 Preparing data for LSTM...")
        
        # Select the 13 core indicators + close price
        feature_columns = [
            'Close', 'RSI', 'ROC', 'Stoch_K', 'Stoch_D', 'TSI',
            'OBV', 'MFI', 'PVT', 'TEMA', 'MACD', 'KAMA',
            'ATR', 'BB_Position', 'Ulcer_Index'
        ]
        
        # Check which features are available
        available_features = []
        for col in feature_columns:
            if col in df.columns:
                available_features.append(col)
            else:
                print(f"⚠️ Missing feature: {col}")
        
        if len(available_features) < 5:
            raise ValueError(f"Need at least 5 features, only found {len(available_features)}")
        
        print(f"📊 Using {len(available_features)} features: {available_features}")
        
        # Extract feature data
        data = df[available_features].copy()
        
        # Handle missing values
        data = data.fillna(method='ffill').fillna(method='bfill')
        
        # Remove rows with any remaining NaN
        data = data.dropna()
        
        print(f"📊 Data shape after cleaning: {data.shape}")
        
        # Scale the data
        scaled_data = self.scaler.fit_transform(data)
        
        # Create sequences
        X, y = self._create_sequences(scaled_data)
        
        # Split data
        split_idx = int(len(X) * (1 - self.test_split))
        
        self.X_train = X[:split_idx]
        self.X_test = X[split_idx:]
        self.y_train = y[:split_idx]
        self.y_test = y[split_idx:]
        
        # Store for inverse transformation
        self.feature_columns = available_features
        self.original_data = df[available_features[0]].values  # Close prices
        
        print(f"✅ Data preparation complete:")
        print(f"   Training samples: {len(self.X_train)}")
        print(f"   Test samples: {len(self.X_test)}")
        print(f"   Feature dimensions: {self.X_train.shape[2]}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def _create_sequences(self, data):
        """Create sequences for LSTM"""
        X, y = [], []
        
        for i in range(self.sequence_length, len(data) - self.target_horizon):
            X.append(data[i - self.sequence_length:i])
            y.append(data[i + self.target_horizon, 0])  # Predict close price
        
        return np.array(X), np.array(y)
    
    def build_model(self, lstm_units=[100, 50], dropout_rate=0.2):
        """Build LSTM model"""
        print("🏗️ Building LSTM model...")
        
        self.model = Sequential([
            LSTM(lstm_units[0], return_sequences=True, 
                 input_shape=(self.X_train.shape[1], self.X_train.shape[2])),
            Dropout(dropout_rate),
            
            LSTM(lstm_units[1], return_sequences=False),
            Dropout(dropout_rate),
            
            Dense(50, activation='relu'),
            Dropout(dropout_rate),
            
            Dense(1)
        ])
        
        self.model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        print("✅ Model built successfully")
        self.model.summary()
        
        return self.model
    
    def train_model(self, epochs=50, batch_size=32, validation_split=0.2):
        """Train the LSTM model"""
        print("🎓 Training LSTM model...")
        
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(factor=0.5, patience=5, verbose=1)
        ]
        
        self.history = self.model.fit(
            self.X_train, self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )
        
        print("✅ Model training completed!")
        return self.history
    
    def make_predictions(self):
        """Make predictions and evaluate"""
        print("🔮 Making predictions...")
        
        # Predict on test data
        y_pred_scaled = self.model.predict(self.X_test, verbose=0)
        
        # Inverse transform predictions
        y_pred = self._inverse_transform_predictions(y_pred_scaled.flatten())
        y_actual = self._inverse_transform_predictions(self.y_test)
        
        self.y_pred = y_pred
        self.y_actual = y_actual
        
        # Calculate metrics
        mae = mean_absolute_error(y_actual, y_pred)
        mse = mean_squared_error(y_actual, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_actual, y_pred)
        
        # Directional accuracy
        actual_direction = np.diff(y_actual) > 0
        pred_direction = np.diff(y_pred) > 0
        directional_accuracy = np.mean(actual_direction == pred_direction) * 100
        
        self.metrics = {
            'MAE': round(mae, 4),
            'MSE': round(mse, 4),
            'RMSE': round(rmse, 4),
            'R2_Score': round(r2, 4),
            'Directional_Accuracy': round(directional_accuracy, 2)
        }
        
        print("📊 Prediction Metrics:")
        for metric, value in self.metrics.items():
            print(f"   {metric}: {value}")
        
        return y_pred, y_actual
    
    def _inverse_transform_predictions(self, scaled_values):
        """Inverse transform scaled predictions back to original price scale"""
        if np.isscalar(scaled_values):
            scaled_values = [scaled_values]
        
        # Create dummy array for inverse transform
        dummy = np.zeros((len(scaled_values), len(self.feature_columns)))
        dummy[:, 0] = scaled_values  # First feature is Close price
        
        # Inverse transform and return only the price column
        return self.scaler.inverse_transform(dummy)[:, 0]
    
    def generate_trading_signals(self, df):
        """Generate trading signals"""
        print("🎯 Generating trading signals...")
        
        if not hasattr(self, 'y_pred'):
            raise ValueError("Must make predictions first")
        
        # Get test period close prices
        close_prices = df['Close'].values
        test_start_idx = len(close_prices) - len(self.y_actual)
        test_close_prices = close_prices[test_start_idx:]
        
        signals = []
        
        for i in range(min(len(self.y_pred), len(test_close_prices)) - self.target_horizon):
            current_price = test_close_prices[i]
            predicted_price = self.y_pred[i]
            
            if i + self.target_horizon < len(test_close_prices):
                future_price = test_close_prices[i + self.target_horizon]
            else:
                continue
            
            # Generate signal
            predicted_change = (predicted_price - current_price) / current_price
            actual_change = (future_price - current_price) / current_price
            
            signal_type = 'BUY' if predicted_change > 0.01 else 'SELL'
            actual_trend = 'BUY' if actual_change > 0 else 'SELL'
            
            confidence = min(abs(predicted_change) * 100, 10)  # Cap at 10%
            
            signals.append({
                'Day': i,
                'Current_Price': round(current_price, 2),
                'Predicted_Price': round(predicted_price, 2),
                'Actual_Future_Price': round(future_price, 2),
                'Signal': signal_type,
                'Confidence': round(confidence, 2),
                'Predicted_Change_%': round(predicted_change * 100, 2),
                'Actual_Change_%': round(actual_change * 100, 2),
                'Correct_Direction': signal_type == actual_trend
            })
        
        self.signals_df = pd.DataFrame(signals)
        
        # Calculate summary
        total_signals = len(self.signals_df)
        correct_signals = self.signals_df['Correct_Direction'].sum()
        accuracy = (correct_signals / total_signals) * 100 if total_signals > 0 else 0
        
        print(f"✅ Generated {total_signals} trading signals")
        print(f"📊 Signal Accuracy: {accuracy:.2f}%")
        
        return self.signals_df
    
    def plot_results(self):
        """Create comprehensive plots"""
        print("📈 Creating visualizations...")
        
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('LSTM Trading Strategy Results', fontsize=16, fontweight='bold')
        
        # 1. Predictions vs Actual
        sample_size = min(100, len(self.y_actual))
        axes[0, 0].plot(self.y_actual[:sample_size], label='Actual', alpha=0.8)
        axes[0, 0].plot(self.y_pred[:sample_size], label='Predicted', alpha=0.8)
        axes[0, 0].set_title('Price Predictions vs Actual')
        axes[0, 0].set_xlabel('Days')
        axes[0, 0].set_ylabel('Price')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Training History
        if self.history:
            axes[0, 1].plot(self.history.history['loss'], label='Training Loss')
            if 'val_loss' in self.history.history:
                axes[0, 1].plot(self.history.history['val_loss'], label='Validation Loss')
            axes[0, 1].set_title('Model Training History')
            axes[0, 1].set_xlabel('Epochs')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Signal Performance
        if hasattr(self, 'signals_df'):
            signal_accuracy = self.signals_df.groupby('Signal')['Correct_Direction'].mean() * 100
            signal_accuracy.plot(kind='bar', ax=axes[1, 0], color=['red', 'green'])
            axes[1, 0].set_title('Signal Accuracy by Type')
            axes[1, 0].set_ylabel('Accuracy (%)')
            axes[1, 0].set_xticklabels(signal_accuracy.index, rotation=0)
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Return Distribution
        if hasattr(self, 'signals_df'):
            axes[1, 1].hist(self.signals_df['Actual_Change_%'], bins=20, alpha=0.7, color='blue')
            axes[1, 1].axvline(0, color='red', linestyle='--', label='Break-even')
            axes[1, 1].set_title('Distribution of Actual Returns')
            axes[1, 1].set_xlabel('Return (%)')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return fig

def load_sample_data():
    """Load or create sample stock data for demonstration"""
    print("📊 Loading sample data...")
    
    # Create sample data if no data is provided
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
    dates = dates[dates.weekday < 5]  # Remove weekends
    
    # Generate realistic stock price data
    np.random.seed(42)
    n_days = len(dates)
    
    # Start with base price and add random walk
    base_price = 100
    daily_returns = np.random.normal(0.0005, 0.02, n_days)  # 0.05% daily return, 2% volatility
    prices = [base_price]
    
    for i in range(1, n_days):
        new_price = prices[-1] * (1 + daily_returns[i])
        prices.append(new_price)
    
    # Create OHLCV data
    df = pd.DataFrame({
        'Date': dates,
        'Open': [p * (1 + np.random.normal(0, 0.005)) for p in prices],
        'High': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'Low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'Close': prices,
        'Volume': np.random.randint(100000, 1000000, n_days)
    })
    
    # Ensure High >= Close >= Low and High >= Open >= Low
    df['High'] = df[['Open', 'High', 'Close']].max(axis=1)
    df['Low'] = df[['Open', 'Low', 'Close']].min(axis=1)
    
    print(f"✅ Created sample data: {len(df)} days from {df['Date'].min().date()} to {df['Date'].max().date()}")
    
    return df

def run_kaggle_lstm_strategy(df=None):
    """Main function to run the complete LSTM strategy on Kaggle"""
    print("\n🚀 STARTING KAGGLE LSTM TRADING STRATEGY")
    print("=" * 60)
    
    # Load data
    if df is None:
        df = load_sample_data()
    
    # Calculate technical indicators
    tech_indicators = TechnicalIndicators(df)
    df_with_indicators = tech_indicators.calculate_all_indicators()
    
    # Initialize and run LSTM strategy
    strategy = KaggleLSTMTradingStrategy(
        sequence_length=60,
        target_horizon=5,
        test_split=0.2
    )
    
    # Prepare data
    X_train, X_test, y_train, y_test = strategy.prepare_data(df_with_indicators)
    
    # Build and train model
    model = strategy.build_model(lstm_units=[100, 50], dropout_rate=0.2)
    history = strategy.train_model(epochs=50, batch_size=32, validation_split=0.2)
    
    # Make predictions
    y_pred, y_actual = strategy.make_predictions()
    
    # Generate trading signals
    signals_df = strategy.generate_trading_signals(df_with_indicators)
    
    # Create visualizations
    fig = strategy.plot_results()
    
    # Display results summary
    print("\n📊 FINAL RESULTS SUMMARY")
    print("=" * 40)
    print("Model Performance:")
    for metric, value in strategy.metrics.items():
        print(f"  {metric}: {value}")
    
    if hasattr(strategy, 'signals_df'):
        total_signals = len(strategy.signals_df)
        correct_signals = strategy.signals_df['Correct_Direction'].sum()
        signal_accuracy = (correct_signals / total_signals) * 100 if total_signals > 0 else 0
        
        print(f"\nTrading Signals:")
        print(f"  Total Signals: {total_signals}")
        print(f"  Correct Predictions: {correct_signals}")
        print(f"  Signal Accuracy: {signal_accuracy:.2f}%")
        
        # Show sample signals
        print(f"\nSample Trading Signals:")
        display_cols = ['Day', 'Signal', 'Confidence', 'Predicted_Change_%', 
                       'Actual_Change_%', 'Correct_Direction']
        print(strategy.signals_df[display_cols].head(10).to_string(index=False))
    
    print(f"\n✅ Kaggle LSTM Strategy completed successfully!")
    
    return strategy, signals_df

# Run the strategy (this will execute when the script is run)
if __name__ == "__main__":
    # For Kaggle, you can load your own data here
    # df = pd.read_csv('/kaggle/input/your-dataset/data.csv')
    
    # Run with sample data for demonstration
    strategy, signals = run_kaggle_lstm_strategy()
