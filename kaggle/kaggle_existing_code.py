#!/usr/bin/env python3
"""
Kaggle LSTM Trading Strategy - Using Existing Codebase

This script adapts your existing run_focused_lstm.py to work in Kaggle environment
by importing your existing modules and using your processed data.

🔧 Setup Required:
1. Upload your processed data CSV to Kaggle dataset
2. Upload your project files (src folder) to Kaggle dataset or copy the code below
3. Update DATA_PATH to point to your processed data file
4. Run this script directly in Kaggle

📁 Expected Data Format:
Your processed CSV should already contain the 13 indicators:
RSI, ROC, Stoch_K, Stoch_D, TSI, OBV, MFI, PVT, TEMA, MACD, MACD_Signal, 
MACD_Histogram, KAMA, ATR, BB_Upper, BB_Middle, BB_Lower, BB_Width, 
BB_Position, Ulcer_Index
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Install required packages for Kaggle
try:
    import tensorflow as tf
    from sklearn.preprocessing import MinMaxScaler
    print("✅ Required packages already available")
except ImportError:
    print("📦 Installing required packages...")
    import subprocess
    packages = ['tensorflow==2.13.0', 'scikit-learn']
    for package in packages:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
    import tensorflow as tf
    from sklearn.preprocessing import MinMaxScaler

# ============================================================================
# CONFIGURATION - UPDATE THESE PATHS FOR YOUR KAGGLE SETUP
# ============================================================================

# Path to your processed data (with indicators already calculated)
DATA_PATH = '/kaggle/input/your-dataset/processed_data_with_indicators.csv'

# Alternative paths if you upload your src files as a dataset
SRC_PATH = '/kaggle/input/your-src-files'  # If you upload src folder
# OR copy the classes below if you can't upload as dataset

# Output directory
OUTPUT_DIR = '/kaggle/working'

# ============================================================================
# OPTION 1: IMPORT YOUR EXISTING MODULES (if uploaded as Kaggle dataset)
# ============================================================================

# Uncomment these if you uploaded your src folder as a Kaggle dataset
# sys.path.insert(0, SRC_PATH)
# from lstm_trading_strategy import LSTMTradingStrategy
# from technical_indicators import *
# from config.lstm_settings import LSTM_CONFIG

# ============================================================================
# OPTION 2: EMBEDDED CLASSES (if you can't upload src files)
# ============================================================================

# Your existing LSTMTradingStrategy class (simplified for Kaggle)
class LSTMTradingStrategy:
    """
    Simplified version of your LSTMTradingStrategy class for Kaggle.
    Contains the core functionality from your original lstm_trading_strategy.py
    """
    
    def __init__(self, sequence_length=60, target_horizon=5, 
                 use_advanced_preprocessing=True, outlier_threshold=2.0, 
                 pca_variance_threshold=0.98):
        self.sequence_length = sequence_length
        self.target_horizon = target_horizon
        self.use_advanced_preprocessing = use_advanced_preprocessing
        self.outlier_threshold = outlier_threshold
        self.pca_variance_threshold = pca_variance_threshold
        
        self.scaler = MinMaxScaler()
        self.model = None
        self.feature_columns = []
        
        print(f"🤖 LSTM Strategy initialized:")
        print(f"   Sequence Length: {sequence_length}")
        print(f"   Target Horizon: {target_horizon}")
        print(f"   Advanced Preprocessing: {use_advanced_preprocessing}")
    
    def prepare_data(self, df):
        """Prepare data for LSTM (adapted from your original)"""
        print("📋 Preparing data for LSTM...")
        
        # Your 13 focused indicators
        self.feature_columns = [
            'RSI', 'ROC', 'Stoch_K', 'Stoch_D', 'TSI',  # Momentum
            'OBV', 'MFI', 'PVT',                         # Volume  
            'TEMA', 'MACD', 'MACD_Signal', 'MACD_Histogram', 'KAMA',  # Trend
            'ATR', 'BB_Width', 'BB_Position', 'Ulcer_Index'  # Volatility
        ]
        
        # Check which indicators are available in your data
        available_features = [col for col in self.feature_columns if col in df.columns]
        missing_features = [col for col in self.feature_columns if col not in df.columns]
        
        if missing_features:
            print(f"⚠️  Missing indicators: {missing_features}")
            print(f"✅ Available indicators: {available_features}")
        else:
            print(f"✅ All {len(self.feature_columns)} indicators found in data")
        
        # Use available features
        self.feature_columns = available_features
        
        # Prepare feature matrix
        feature_data = df[self.feature_columns + ['Close']].copy()
        feature_data = feature_data.dropna()
        
        if len(feature_data) < self.sequence_length + self.target_horizon:
            raise ValueError(f"Not enough data. Need at least {self.sequence_length + self.target_horizon} rows")
        
        # Scale features
        scaled_data = self.scaler.fit_transform(feature_data)
        
        # Create sequences
        X, y = [], []
        dates = []
        
        for i in range(self.sequence_length, len(scaled_data) - self.target_horizon):
            X.append(scaled_data[i-self.sequence_length:i, :-1])  # All features except Close
            y.append(scaled_data[i + self.target_horizon, -1])    # Future Close price
            dates.append(feature_data.index[i])
        
        X, y = np.array(X), np.array(y)
        
        print(f"✅ Data prepared: X shape={X.shape}, y shape={y.shape}")
        return X, y, self.feature_columns, dates
    
    def build_model(self, lstm_units=[100, 50], dropout_rate=0.2, optimize_features=False):
        """Build LSTM model (adapted from your original)"""
        print("🏗️ Building LSTM model...")
        
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.optimizers import Adam
        
        # Build model architecture
        self.model = Sequential([
            LSTM(lstm_units[0], return_sequences=True, 
                 input_shape=(self.sequence_length, len(self.feature_columns))),
            Dropout(dropout_rate),
            LSTM(lstm_units[1], return_sequences=False),
            Dropout(dropout_rate),
            Dense(50, activation='relu'),
            Dense(25, activation='relu'),
            Dense(1)
        ])
        
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        print(f"✅ Model built with {sum(p.numel() for p in self.model.parameters())} parameters")
        return self.model
    
    def train_model(self, X, y, epochs=50, batch_size=32, validation_split=0.2):
        """Train the LSTM model"""
        print("🎓 Training LSTM model...")
        
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
        
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(factor=0.5, patience=5, verbose=1)
        ]
        
        history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )
        
        print("✅ Training completed!")
        return history
    
    def generate_trading_signals(self, df):
        """Generate trading signals (adapted from your original)"""
        print("📈 Generating trading signals...")
        
        # Prepare data for prediction
        feature_data = df[self.feature_columns + ['Close']].copy()
        feature_data = feature_data.dropna()
        
        # Scale data
        scaled_data = self.scaler.transform(feature_data)
        
        # Generate predictions
        predictions = []
        signals = []
        confidences = []
        
        for i in range(self.sequence_length, len(scaled_data)):
            # Prepare sequence
            sequence = scaled_data[i-self.sequence_length:i, :-1].reshape(1, self.sequence_length, -1)
            
            # Make prediction
            pred_scaled = self.model.predict(sequence, verbose=0)[0, 0]
            
            # Inverse transform to get actual price prediction
            dummy_pred = np.zeros((1, scaled_data.shape[1]))
            dummy_pred[0, -1] = pred_scaled
            pred_price = self.scaler.inverse_transform(dummy_pred)[0, -1]
            
            predictions.append(pred_price)
            
            # Generate signal
            current_price = feature_data['Close'].iloc[i]
            price_change = (pred_price - current_price) / current_price
            
            if price_change > 0.02:  # 2% threshold for buy
                signal = 1
                confidence = min(abs(price_change) * 100, 10)
            elif price_change < -0.02:  # -2% threshold for sell
                signal = -1
                confidence = min(abs(price_change) * 100, 10)
            else:
                signal = 0
                confidence = 0
            
            signals.append(signal)
            confidences.append(confidence)
        
        # Create results dataframe
        start_idx = self.sequence_length
        results_df = feature_data.iloc[start_idx:start_idx+len(predictions)].copy()
        results_df['Prediction'] = predictions
        results_df['Signal'] = signals
        results_df['Confidence'] = confidences
        
        print(f"✅ Generated {len(results_df)} trading signals")
        return results_df
    
    def evaluate_strategy(self, df, signals_df):
        """Evaluate strategy performance (simplified version)"""
        print("📊 Evaluating strategy performance...")
        
        # Calculate prediction accuracy
        actual_prices = signals_df['Close'].values
        predicted_prices = signals_df['Prediction'].values
        
        mae = np.mean(np.abs(actual_prices - predicted_prices))
        mse = np.mean((actual_prices - predicted_prices) ** 2)
        rmse = np.sqrt(mse)
        
        # Calculate directional accuracy
        actual_changes = np.diff(actual_prices)
        predicted_changes = np.diff(predicted_prices)
        directional_accuracy = np.mean(np.sign(actual_changes) == np.sign(predicted_changes)) * 100
        
        # Simple portfolio simulation
        initial_capital = 100000
        capital = initial_capital
        position = 0
        
        for i, signal in enumerate(signals_df['Signal']):
            price = signals_df['Close'].iloc[i]
            
            if signal == 1 and position <= 0:  # Buy
                shares = capital // price
                capital -= shares * price
                position = shares
            elif signal == -1 and position > 0:  # Sell
                capital += position * price
                position = 0
        
        # Final portfolio value
        if position > 0:
            capital += position * signals_df['Close'].iloc[-1]
        
        total_return = (capital - initial_capital) / initial_capital * 100
        
        evaluation_results = {
            'accuracy_metrics': {
                'mae': round(mae, 2),
                'rmse': round(rmse, 2),
                'directional_accuracy': round(directional_accuracy, 2)
            },
            'portfolio_performance': {
                'initial_capital': initial_capital,
                'final_capital': round(capital, 2),
                'total_return_pct': round(total_return, 2),
                'total_signals': len(signals_df),
                'buy_signals': len(signals_df[signals_df['Signal'] == 1]),
                'sell_signals': len(signals_df[signals_df['Signal'] == -1])
            },
            'signal_quality': {
                'average_confidence': round(signals_df['Confidence'].mean(), 2),
                'high_confidence_signals': len(signals_df[signals_df['Confidence'] > 5])
            }
        }
        
        return evaluation_results

# ============================================================================
# MAIN EXECUTION FUNCTION
# ============================================================================

def run_kaggle_focused_lstm():
    """Main function to run the focused LSTM strategy in Kaggle"""
    print("🎯 KAGGLE FOCUSED LSTM TRADING STRATEGY")
    print("=" * 60)
    print("Using existing codebase and processed data")
    print()
    
    try:
        # 1. Load your processed data
        print("📂 Loading processed data with indicators...")
        df = pd.read_csv(DATA_PATH)
        
        # Handle date column
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
        elif not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        print(f"✅ Data loaded: {df.shape}")
        print(f"📅 Date range: {df.index.min()} to {df.index.max()}")
        print(f"📊 Available columns: {list(df.columns)}")
        
        # 2. Initialize LSTM strategy with your settings
        strategy = LSTMTradingStrategy(
            sequence_length=60,                  # From your original settings
            target_horizon=5,                    # From your original settings
            use_advanced_preprocessing=True,     # From your original settings
            outlier_threshold=2.0,              # From your original settings
            pca_variance_threshold=0.98         # From your original settings
        )
        
        # 3. Prepare data
        X, y, feature_columns, dates = strategy.prepare_data(df)
        
        # 4. Build model
        strategy.build_model(
            lstm_units=[100, 50],               # From your original settings
            dropout_rate=0.2,                   # From your original settings
            optimize_features=False             # Disabled for speed in Kaggle
        )
        
        # 5. Train model
        history = strategy.train_model(
            X, y, 
            epochs=30,                          # Reduced for Kaggle
            batch_size=32, 
            validation_split=0.2
        )
        
        # 6. Generate trading signals
        signals_df = strategy.generate_trading_signals(df)
        
        # 7. Evaluate strategy
        evaluation_results = strategy.evaluate_strategy(df, signals_df)
        
        # 8. Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save signals
        signals_file = f'{OUTPUT_DIR}/kaggle_focused_signals_{timestamp}.csv'
        signals_df.to_csv(signals_file)
        print(f"💾 Signals saved: {signals_file}")
        
        # Save evaluation report
        report_file = f'{OUTPUT_DIR}/kaggle_focused_report_{timestamp}.txt'
        with open(report_file, 'w') as f:
            f.write("KAGGLE FOCUSED LSTM TRADING STRATEGY REPORT\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Data source: {DATA_PATH}\n")
            f.write(f"Data period: {df.index.min()} to {df.index.max()}\n\n")
            
            f.write("INDICATORS USED:\n")
            f.write("-" * 20 + "\n")
            for i, indicator in enumerate(feature_columns, 1):
                f.write(f"{i:2d}. {indicator}\n")
            f.write(f"\nTotal indicators: {len(feature_columns)}\n\n")
            
            f.write("PERFORMANCE METRICS:\n")
            f.write("-" * 25 + "\n")
            for category, metrics in evaluation_results.items():
                f.write(f"\n{category.upper()}:\n")
                if isinstance(metrics, dict):
                    for key, value in metrics.items():
                        f.write(f"  {key}: {value}\n")
                else:
                    f.write(f"  {metrics}\n")
        
        print(f"📄 Report saved: {report_file}")
        
        # 9. Display summary
        print("\n" + "=" * 50)
        print("🎯 KAGGLE STRATEGY SUMMARY")
        print("=" * 50)
        
        accuracy_metrics = evaluation_results.get('accuracy_metrics', {})
        portfolio_performance = evaluation_results.get('portfolio_performance', {})
        signal_quality = evaluation_results.get('signal_quality', {})
        
        print(f"📊 Model Accuracy:")
        print(f"   Directional Accuracy: {accuracy_metrics.get('directional_accuracy', 'N/A')}%")
        print(f"   MAE: {accuracy_metrics.get('mae', 'N/A')}")
        print(f"   RMSE: {accuracy_metrics.get('rmse', 'N/A')}")
        
        print(f"\n💰 Portfolio Performance:")
        print(f"   Total Return: {portfolio_performance.get('total_return_pct', 'N/A')}%")
        print(f"   Final Capital: ${portfolio_performance.get('final_capital', 'N/A'):,.2f}")
        
        print(f"\n🎯 Signal Quality:")
        print(f"   Total Signals: {portfolio_performance.get('total_signals', 'N/A')}")
        print(f"   Buy Signals: {portfolio_performance.get('buy_signals', 'N/A')}")
        print(f"   Sell Signals: {portfolio_performance.get('sell_signals', 'N/A')}")
        print(f"   Average Confidence: {signal_quality.get('average_confidence', 'N/A')}")
        
        print("\n✅ Kaggle strategy completed successfully!")
        print(f"📁 Results saved to: {OUTPUT_DIR}")
        
        # Create a simple visualization
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Price vs Predictions
        sample_size = min(200, len(signals_df))
        axes[0, 0].plot(signals_df['Close'].iloc[:sample_size], label='Actual', alpha=0.8)
        axes[0, 0].plot(signals_df['Prediction'].iloc[:sample_size], label='Predicted', alpha=0.8)
        axes[0, 0].set_title('Price Predictions vs Actual')
        axes[0, 0].legend()
        
        # Training history
        axes[0, 1].plot(history.history['loss'], label='Training Loss')
        if 'val_loss' in history.history:
            axes[0, 1].plot(history.history['val_loss'], label='Validation Loss')
        axes[0, 1].set_title('Training History')
        axes[0, 1].legend()
        
        # Signal distribution
        signal_counts = signals_df['Signal'].value_counts()
        axes[1, 0].bar(['Sell (-1)', 'Hold (0)', 'Buy (1)'], 
                      [signal_counts.get(-1, 0), signal_counts.get(0, 0), signal_counts.get(1, 0)])
        axes[1, 0].set_title('Signal Distribution')
        
        # Confidence distribution
        axes[1, 1].hist(signals_df['Confidence'], bins=20, alpha=0.7)
        axes[1, 1].set_title('Confidence Distribution')
        axes[1, 1].set_xlabel('Confidence Level')
        
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/kaggle_analysis_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error running Kaggle strategy: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

# ============================================================================
# EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("🚀 Starting Kaggle LSTM Trading Strategy...")
    print("=" * 60)
    print(f"📍 Data Path: {DATA_PATH}")
    print(f"📁 Output Directory: {OUTPUT_DIR}")
    print("=" * 60)
    
    success = run_kaggle_focused_lstm()
    
    if success:
        print("\n🎉 SUCCESS! All files generated in /kaggle/working/")
        print("📊 Check the visualization above and download the CSV files")
    else:
        print("\n❌ FAILED! Check the error messages above")
