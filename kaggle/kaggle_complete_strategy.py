#!/usr/bin/env python3
"""
Kaggle LSTM Trading Strategy - Using Your Existing Codebase

This script brings your complete existing LSTM trading strategy to Kaggle.
It includes your LSTMTradingStrategy class, settings, and runs the focused strategy
on your processed data with indicators already calculated.

🔧 Setup Instructions:
1. Upload your processed CSV file with indicators to Kaggle dataset
2. Update DATA_PATH below to match your uploaded data location
3. Run this script in a Kaggle notebook cell
4. All outputs will be saved to /kaggle/working/

📊 Expected Data Format:
Your CSV should contain OHLCV data plus the 13 focused indicators:
RSI, ROC, Stoch_K, Stoch_D, TSI, OBV, MFI, PVT, TEMA, MACD, 
MACD_Signal, MACD_Histogram, KAMA, ATR, BB_Width, BB_Position, Ulcer_Index
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Install required packages for Kaggle
print("📦 Checking/Installing required packages...")
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.decomposition import PCA
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from scipy import stats
    import matplotlib.pyplot as plt
    import seaborn as sns
    print("✅ All required packages available")
except ImportError:
    import subprocess
    packages = [
        'tensorflow==2.13.0',
        'scikit-learn',
        'scipy',
        'matplotlib',
        'seaborn'
    ]
    for package in packages:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
    
    # Re-import after installation
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.decomposition import PCA
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from scipy import stats
    import matplotlib.pyplot as plt
    import seaborn as sns

# ============================================================================
# CONFIGURATION - UPDATE THIS PATH FOR YOUR DATA
# ============================================================================

# 🔴 UPDATE THIS PATH TO YOUR KAGGLE DATASET
DATA_PATH = r'C:\Users\98765\OneDrive\Desktop\Timesnow\data\processed\stock_data_with_technical_indicators.csv'

OUTPUT_DIR = r'C:\Users\98765\OneDrive\Desktop\Timesnow\output'

# Company selection (if your dataset has multiple companies)
COMPANY_SELECTION = {
    'method': 'most_data',  # 'most_data', 'specific_id', or 'random'
    'company_id': None,     # Set specific company ID if method is 'specific_id'
    'sample_size': 5000     # Maximum number of records to use (for faster processing)
}

# Your LSTM settings (from config/lstm_settings.py)
LSTM_CONFIG = {
    'sequence_length': 60,
    'target_horizon': 5,
    'target_gain': 0.10,
    'stop_loss': -0.03,
    'test_split': 0.2,
    'use_advanced_preprocessing': True,
    'outlier_threshold': 2.0,
    'pca_variance_threshold': 0.98
}

# Model architecture settings (separate from init parameters)
MODEL_CONFIG = {
    'lstm_units': [100, 50],
    'dropout_rate': 0.2,
    'epochs': 50,  # Reduced for Kaggle
    'batch_size': 32,
    'validation_split': 0.2
}

# Your 13 focused indicators
FOCUSED_INDICATORS = [
    'RSI', 'ROC', 'Stoch_K', 'Stoch_D', 'TSI',  # Momentum
    'OBV', 'MFI', 'PVT',                         # Volume
    'TEMA', 'MACD', 'MACD_Signal', 'MACD_Histogram', 'KAMA',  # Trend
    'ATR', 'BB_Width', 'BB_Position', 'Ulcer_Index'  # Volatility
]

print("🎯 KAGGLE LSTM TRADING STRATEGY")
print("=" * 60)
print("Using your existing codebase with processed data")
print(f"📁 Data path: {DATA_PATH}")
print(f"📊 Using {len(FOCUSED_INDICATORS)} focused indicators")
print("=" * 60)

# ============================================================================
# YOUR LSTM TRADING STRATEGY CLASS
# ============================================================================

class LSTMTradingStrategy:
    """
    Your complete LSTM Trading Strategy class adapted for Kaggle.
    Based on your src/lstm_trading_strategy.py with key functionality preserved.
    """
    
    def __init__(self, 
                 sequence_length=60,
                 target_horizon=5,
                 target_gain=0.10,
                 stop_loss=-0.03,
                 test_split=0.2,
                 use_advanced_preprocessing=True,
                 outlier_threshold=1.5,
                 pca_variance_threshold=0.95):
        """Initialize LSTM Trading Strategy with your original parameters"""
        
        self.sequence_length = sequence_length
        self.target_horizon = target_horizon  
        self.target_gain = target_gain
        self.stop_loss = stop_loss
        self.test_split = test_split
        self.use_advanced_preprocessing = use_advanced_preprocessing
        self.outlier_threshold = outlier_threshold
        self.pca_variance_threshold = pca_variance_threshold
        
        # Initialize components
        self.scaler = MinMaxScaler()
        self.pca = None
        self.model = None
        self.feature_columns = []
        self.training_history = None
        
        print(f"🤖 LSTM Strategy initialized:")
        print(f"   Sequence Length: {sequence_length}")
        print(f"   Target Horizon: {target_horizon}")
        print(f"   Target Gain: {target_gain*100:.1f}%")
        print(f"   Stop Loss: {stop_loss*100:.1f}%")
        print(f"   Advanced Preprocessing: {use_advanced_preprocessing}")

    def prepare_data(self, df):
        """Prepare data for LSTM training (your original logic)"""
        print("📋 Preparing data for LSTM...")
        
        # Use your focused indicators
        available_indicators = [col for col in FOCUSED_INDICATORS if col in df.columns]
        missing_indicators = [col for col in FOCUSED_INDICATORS if col not in df.columns]
        
        if missing_indicators:
            print(f"⚠️  Missing indicators: {missing_indicators}")
        
        print(f"✅ Using {len(available_indicators)} indicators: {available_indicators}")
        
        # Prepare feature matrix
        self.feature_columns = available_indicators
        
        # Handle column name variations (close vs Close)
        close_col = None
        if 'Close' in df.columns:
            close_col = 'Close'
        elif 'close' in df.columns:
            close_col = 'close'
        else:
            raise ValueError("Neither 'Close' nor 'close' column found in data")
        
        feature_data = df[self.feature_columns + [close_col]].copy()
        
        # Rename close column to standardize
        if close_col == 'close':
            feature_data = feature_data.rename(columns={'close': 'Close'})
        
        # Handle missing values
        feature_data = feature_data.fillna(method='ffill').fillna(method='bfill')
        feature_data = feature_data.dropna()
        
        if len(feature_data) < self.sequence_length + self.target_horizon + 100:
            raise ValueError(f"Insufficient data. Need at least {self.sequence_length + self.target_horizon + 100} rows")
        
        # Advanced preprocessing (if enabled)
        if self.use_advanced_preprocessing:
            feature_data = self._remove_outliers(feature_data)
            print(f"   📊 After outlier removal: {len(feature_data)} rows")
        
        # Scale features
        scaled_features = self.scaler.fit_transform(feature_data)
        
        # Apply PCA if enabled
        if self.use_advanced_preprocessing and len(self.feature_columns) > 10:
            self.pca = PCA(n_components=self.pca_variance_threshold)
            # Extract feature matrix (excluding Close price column)
            feature_matrix = scaled_features[:, :-1]
            # Apply PCA transformation
            pca_features = self.pca.fit_transform(feature_matrix)
            # Reconstruct scaled_features with PCA features + Close price
            scaled_features = np.column_stack([pca_features, scaled_features[:, -1]])
            print(f"   🔧 PCA applied: {self.pca.n_components_} components (reduced from {len(self.feature_columns)} features)")
            # Update feature columns count for model building
            self.n_features_after_pca = self.pca.n_components_
        else:
            self.n_features_after_pca = len(self.feature_columns)
        
        # Create sequences for LSTM
        X, y = [], []
        dates = []
        
        for i in range(self.sequence_length, len(scaled_features) - self.target_horizon):
            X.append(scaled_features[i-self.sequence_length:i, :-1])  # Features (now properly sized)
            y.append(scaled_features[i + self.target_horizon, -1])    # Future Close
            dates.append(feature_data.index[i])
        
        X, y = np.array(X), np.array(y)
        
        print(f"✅ Data prepared: X shape={X.shape}, y shape={y.shape}")
        print(f"   📊 Feature dimension: {X.shape[2]}")
        print(f"   📅 Date range: {dates[0]} to {dates[-1]}")
        
        return X, y, self.feature_columns, dates
    
    def _remove_outliers(self, df):
        """Remove outliers using Z-score method"""
        df_clean = df.copy()
        
        for column in self.feature_columns:
            if column in df_clean.columns:
                z_scores = np.abs(stats.zscore(df_clean[column]))
                df_clean = df_clean[z_scores < self.outlier_threshold]
        
        return df_clean
    
    def build_model(self, lstm_units=[100, 50], dropout_rate=0.2, optimize_features=True):
        """Build LSTM model (your original architecture)"""
        print("🏗️ Building LSTM model...")
        
        input_shape = (self.sequence_length, self.n_features_after_pca)
        
        self.model = Sequential([
            LSTM(lstm_units[0], 
                 return_sequences=True, 
                 input_shape=input_shape,
                 dropout=dropout_rate,
                 recurrent_dropout=dropout_rate),
            
            LSTM(lstm_units[1], 
                 return_sequences=False,
                 dropout=dropout_rate,
                 recurrent_dropout=dropout_rate),
            
            Dense(50, activation='relu'),
            Dropout(dropout_rate),
            Dense(25, activation='relu'),
            Dropout(dropout_rate),
            Dense(1)
        ])
        
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        print(f"✅ Model built successfully")
        print(f"   📊 Total parameters: {self.model.count_params():,}")
        
        return self.model
    
    def train_model(self, X, y, epochs=50, batch_size=32, validation_split=0.2):
        """Train the LSTM model (your original training logic)"""
        print("🎓 Training LSTM model...")
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                patience=10, 
                restore_best_weights=True, 
                verbose=1,
                monitor='val_loss'
            ),
            ReduceLROnPlateau(
                factor=0.5, 
                patience=5, 
                verbose=1,
                monitor='val_loss'
            )
        ]
        
        # Train model
        self.training_history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )
        
        print("✅ Training completed!")
        return self.training_history
    
    def generate_trading_signals(self, df):
        """Generate trading signals (your original signal logic)"""
        print("📈 Generating trading signals...")
        
        # Prepare data for prediction
        # Handle column name variations (close vs Close)
        close_col = None
        if 'Close' in df.columns:
            close_col = 'Close'
        elif 'close' in df.columns:
            close_col = 'close'
        else:
            raise ValueError("Neither 'Close' nor 'close' column found in data")
        
        feature_data = df[self.feature_columns + [close_col]].copy()
        
        # Rename close column to standardize
        if close_col == 'close':
            feature_data = feature_data.rename(columns={'close': 'Close'})
            
        feature_data = feature_data.fillna(method='ffill').fillna(method='bfill')
        feature_data = feature_data.dropna()
        
        # Apply same preprocessing as training
        if self.use_advanced_preprocessing:
            feature_data = self._remove_outliers(feature_data)
        
        # Scale data
        scaled_data = self.scaler.transform(feature_data)
        
        # Apply PCA if used during training
        if self.pca is not None:
            # Separate features and target
            scaled_features_only = scaled_data[:, :-1]  # All except close price
            scaled_target = scaled_data[:, -1:]         # Close price only
            
            # Apply PCA to features
            pca_features = self.pca.transform(scaled_features_only)
            
            # Reconstruct scaled_data with PCA-transformed features
            scaled_data = np.column_stack([pca_features, scaled_target])
        
        # Generate predictions
        predictions = []
        signals = []
        confidences = []
        
        for i in range(self.sequence_length, len(scaled_data)):
            # Prepare sequence
            if self.pca is not None:
                sequence = scaled_data[i-self.sequence_length:i, :-1]
            else:
                sequence = scaled_data[i-self.sequence_length:i, :-1]
            
            sequence = sequence.reshape(1, self.sequence_length, -1)
            
            # Make prediction
            pred_scaled = self.model.predict(sequence, verbose=0)[0, 0]
            
            # Inverse transform prediction
            if self.pca is not None:
                # For PCA case, we need to handle inverse transform differently
                # Since we only predict the close price, we can inverse transform just that
                # Get the close price column scaling parameters from the original scaler
                close_col_min = self.scaler.min_[-1]  # Last column (close price)
                close_col_scale = self.scaler.scale_[-1]  # Last column scale
                # Manually inverse transform the close price prediction
                pred_price = pred_scaled / close_col_scale + close_col_min
            else:
                # Original method for non-PCA case
                dummy_pred = np.zeros((1, scaled_data.shape[1]))
                dummy_pred[0, -1] = pred_scaled
                pred_price = self.scaler.inverse_transform(dummy_pred)[0, -1]
            
            predictions.append(pred_price)
            
            # Generate signal based on prediction
            current_price = feature_data['Close'].iloc[i]
            price_change = (pred_price - current_price) / current_price
            
            # Signal logic (your original thresholds)
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
        
        # Add signal meanings
        results_df['Signal_Meaning'] = results_df['Signal'].map({
            1: 'BUY', 0: 'HOLD', -1: 'SELL'
        })
        
        print(f"✅ Generated {len(results_df)} trading signals")
        print(f"   🔢 Buy signals: {len(results_df[results_df['Signal'] == 1])}")
        print(f"   🔢 Sell signals: {len(results_df[results_df['Signal'] == -1])}")
        print(f"   🔢 Hold signals: {len(results_df[results_df['Signal'] == 0])}")
        
        return results_df
    
    def evaluate_strategy(self, df, signals_df):
        """Evaluate strategy performance (your original evaluation logic)"""
        print("📊 Evaluating strategy performance...")
        
        # Prediction accuracy metrics
        actual_prices = signals_df['Close'].values
        predicted_prices = signals_df['Prediction'].values
        
        mae = mean_absolute_error(actual_prices, predicted_prices)
        mse = mean_squared_error(actual_prices, predicted_prices)
        rmse = np.sqrt(mse)
        r2 = r2_score(actual_prices, predicted_prices)
        
        # Directional accuracy
        actual_changes = np.diff(actual_prices)
        predicted_changes = np.diff(predicted_prices)
        directional_accuracy = np.mean(np.sign(actual_changes) == np.sign(predicted_changes)) * 100
        
        # Portfolio simulation
        initial_capital = 100000
        capital = initial_capital
        position = 0
        trades = []
        portfolio_values = []
        
        for i, row in signals_df.iterrows():
            price = row['Close']
            signal = row['Signal']
            date = i if hasattr(i, 'date') else i
            
            # Portfolio value tracking
            portfolio_value = capital + (position * price)
            portfolio_values.append(portfolio_value)
            
            # Trading logic
            if signal == 1 and position <= 0:  # Buy signal
                if capital > 0:
                    shares = int(capital * 0.95 // price)  # Use 95% of capital
                    if shares > 0:
                        cost = shares * price
                        capital -= cost
                        position = shares
                        trades.append({
                            'date': date, 'action': 'BUY', 'price': price, 
                            'shares': shares, 'cost': cost
                        })
                        
            elif signal == -1 and position > 0:  # Sell signal
                revenue = position * price
                capital += revenue
                trades.append({
                    'date': date, 'action': 'SELL', 'price': price,
                    'shares': position, 'revenue': revenue
                })
                position = 0
        
        # Final portfolio value
        final_portfolio_value = capital + (position * signals_df['Close'].iloc[-1])
        total_return = (final_portfolio_value - initial_capital) / initial_capital
        
        # Calculate additional metrics
        if len(portfolio_values) > 1:
            returns = pd.Series(portfolio_values).pct_change().dropna()
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
            max_drawdown = (pd.Series(portfolio_values) / pd.Series(portfolio_values).cummax() - 1).min()
        else:
            sharpe_ratio = 0
            max_drawdown = 0
        
        # Compile results
        evaluation_results = {
            'accuracy_metrics': {
                'mae': round(mae, 4),
                'rmse': round(rmse, 4),
                'r2_score': round(r2, 4),
                'directional_accuracy': round(directional_accuracy, 2),
                'overall_accuracy': round(directional_accuracy, 2)
            },
            'portfolio_performance': {
                'initial_capital': initial_capital,
                'final_value': round(final_portfolio_value, 2),
                'total_return_pct': round(total_return * 100, 2),
                'sharpe_ratio': round(sharpe_ratio, 3),
                'max_drawdown_pct': round(max_drawdown * 100, 2),
                'total_trades': len(trades)
            },
            'signal_quality': {
                'total_signals': len(signals_df),
                'buy_signals': len(signals_df[signals_df['Signal'] == 1]),
                'sell_signals': len(signals_df[signals_df['Signal'] == -1]),
                'hold_signals': len(signals_df[signals_df['Signal'] == 0]),
                'average_confidence': round(signals_df['Confidence'].mean(), 2),
                'high_confidence_signals': len(signals_df[signals_df['Confidence'] > 5])
            }
        }
        
        return evaluation_results

# ============================================================================
# MAIN EXECUTION FUNCTION
# ============================================================================

def run_kaggle_focused_lstm():
    """Main function to run your focused LSTM strategy in Kaggle"""
    print("🎯 Running Focused LSTM Trading Strategy in Kaggle")
    print("=" * 60)
    
    try:
        # 1. Load your processed data
        print("📂 Loading your processed data...")
        df = pd.read_csv(DATA_PATH)
        
        # Handle date column
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
        elif not isinstance(df.index, pd.DatetimeIndex):
            try:
                df.index = pd.to_datetime(df.index)
            except:
                df.index = pd.to_datetime(df.iloc[:, 0])
                df = df.iloc[:, 1:]
        
        print(f"✅ Data loaded successfully:")
        print(f"   📊 Shape: {df.shape}")
        print(f"   📅 Date range: {df.index.min()} to {df.index.max()}")
        print(f"   📋 Columns: {list(df.columns)}")
        
        # Handle multi-company dataset
        if 'companyid' in df.columns:
            unique_companies = df['companyid'].nunique()
            print(f"   🏢 Found {unique_companies} companies in dataset")
            
            # Select company based on configuration
            if COMPANY_SELECTION['method'] == 'specific_id' and COMPANY_SELECTION['company_id']:
                selected_company = COMPANY_SELECTION['company_id']
                if selected_company not in df['companyid'].values:
                    print(f"   ⚠️  Company {selected_company} not found, using most data method")
                    selected_company = df['companyid'].value_counts().index[0]
            elif COMPANY_SELECTION['method'] == 'random':
                selected_company = df['companyid'].sample(1).iloc[0]
            else:  # 'most_data' (default)
                company_counts = df['companyid'].value_counts()
                selected_company = company_counts.index[0]
            
            company_data_count = df[df['companyid'] == selected_company].shape[0]
            print(f"   🎯 Selected company {selected_company} with {company_data_count} records")
            
            df = df[df['companyid'] == selected_company].copy()
            
            # Apply sample size limit if specified
            if COMPANY_SELECTION['sample_size'] and len(df) > COMPANY_SELECTION['sample_size']:
                df = df.tail(COMPANY_SELECTION['sample_size'])  # Use most recent data
                print(f"   ✂️  Sampled to {len(df)} most recent records")
            
            # Clean up columns
            columns_to_drop = ['companyid', 'companyName', 'Unnamed: 0']
            df = df.drop([col for col in columns_to_drop if col in df.columns], axis=1)
            
            # Reset and create proper date index
            df = df.reset_index(drop=True)
            df.index = pd.date_range(start='2020-01-01', periods=len(df), freq='D')
            
            print(f"   📊 After processing: {df.shape}")
            print(f"   📅 Date range: {df.index.min()} to {df.index.max()}")
        
        elif len(df) > 10000:  # Large single dataset
            # Sample for faster processing if dataset is very large
            if COMPANY_SELECTION['sample_size'] and len(df) > COMPANY_SELECTION['sample_size']:
                df = df.tail(COMPANY_SELECTION['sample_size'])
                print(f"   ✂️  Sampled to {len(df)} most recent records for faster processing")
                
                # Reset date index
                df = df.reset_index(drop=True)
                df.index = pd.date_range(start='2020-01-01', periods=len(df), freq='D')
        
        # Check for required columns
        close_col = 'Close' if 'Close' in df.columns else 'close' if 'close' in df.columns else None
        if close_col is None:
            raise ValueError("Neither 'Close' nor 'close' column found in data")
        
        required_cols = [close_col] + FOCUSED_INDICATORS
        available_cols = [col for col in required_cols if col in df.columns]
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"⚠️  Missing columns: {missing_cols}")
        print(f"✅ Available columns: {len(available_cols)}/{len(required_cols)}")
        
        # 2. Initialize your LSTM strategy
        strategy = LSTMTradingStrategy(**LSTM_CONFIG)
        
        # 3. Prepare data using your logic
        X, y, feature_columns, dates = strategy.prepare_data(df)
        
        # 4. Build model using your architecture
        strategy.build_model(
            lstm_units=MODEL_CONFIG['lstm_units'],
            dropout_rate=MODEL_CONFIG['dropout_rate']
        )
        
        # 5. Train model
        history = strategy.train_model(
            X, y,
            epochs=MODEL_CONFIG['epochs'],
            batch_size=MODEL_CONFIG['batch_size'],
            validation_split=MODEL_CONFIG['validation_split']
        )
        
        # 6. Generate trading signals
        signals_df = strategy.generate_trading_signals(df)
        
        # 7. Evaluate strategy performance
        evaluation_results = strategy.evaluate_strategy(df, signals_df)
        
        # 8. Save results to Kaggle working directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save signals
        signals_file = f'{OUTPUT_DIR}/kaggle_focused_signals_{timestamp}.csv'
        signals_df.to_csv(signals_file)
        print(f"💾 Signals saved: {signals_file}")
        
        # Save detailed report
        report_file = f'{OUTPUT_DIR}/kaggle_focused_report_{timestamp}.txt'
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("KAGGLE FOCUSED LSTM TRADING STRATEGY REPORT\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Data source: {DATA_PATH}\n")
            f.write(f"Data period: {df.index.min()} to {df.index.max()}\n")
            f.write(f"Total data points: {len(df)}\n\n")
            
            f.write("FOCUSED INDICATORS (13 Total):\n")
            f.write("-" * 30 + "\n")
            for i, indicator in enumerate(FOCUSED_INDICATORS, 1):
                status = "✅" if indicator in df.columns else "❌"
                f.write(f"{i:2d}. {indicator} {status}\n")
            
            f.write(f"\nMODEL CONFIGURATION:\n")
            f.write("-" * 30 + "\n")
            f.write("LSTM Configuration:\n")
            for key, value in LSTM_CONFIG.items():
                f.write(f"  {key}: {value}\n")
            f.write("Model Configuration:\n")
            for key, value in MODEL_CONFIG.items():
                f.write(f"  {key}: {value}\n")
            
            f.write(f"\nPERFORMANCE RESULTS:\n")
            f.write("-" * 30 + "\n")
            for category, metrics in evaluation_results.items():
                f.write(f"\n{category.upper()}:\n")
                if isinstance(metrics, dict):
                    for key, value in metrics.items():
                        f.write(f"  {key}: {value}\n")
                else:
                    f.write(f"  {metrics}\n")
        
        print(f"📄 Report saved: {report_file}")
        
        # 9. Create visualizations
        print("📊 Creating visualizations...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Kaggle LSTM Trading Strategy Results', fontsize=16, fontweight='bold')
        
        # Price vs Predictions
        sample_size = min(200, len(signals_df))
        axes[0, 0].plot(signals_df['Close'].iloc[:sample_size], 
                       label='Actual Price', linewidth=2, alpha=0.8)
        axes[0, 0].plot(signals_df['Prediction'].iloc[:sample_size], 
                       label='LSTM Prediction', linewidth=2, alpha=0.8)
        axes[0, 0].set_title('Price Predictions vs Actual (Sample)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Training History
        axes[0, 1].plot(history.history['loss'], label='Training Loss', linewidth=2)
        if 'val_loss' in history.history:
            axes[0, 1].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
        axes[0, 1].set_title('Model Training History')
        axes[0, 1].set_xlabel('Epochs')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Signal Distribution
        signal_counts = signals_df['Signal'].value_counts().sort_index()
        signal_labels = ['SELL (-1)', 'HOLD (0)', 'BUY (1)']
        colors = ['red', 'gray', 'green']
        
        bars = axes[1, 0].bar(range(len(signal_counts)), signal_counts.values, 
                             color=colors[:len(signal_counts)], alpha=0.7)
        axes[1, 0].set_title('Trading Signal Distribution')
        axes[1, 0].set_xticks(range(len(signal_counts)))
        axes[1, 0].set_xticklabels([signal_labels[i+1] for i in signal_counts.index])
        axes[1, 0].set_ylabel('Number of Signals')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                           f'{int(height)}', ha='center', va='bottom')
        
        # Confidence Distribution
        axes[1, 1].hist(signals_df['Confidence'], bins=20, alpha=0.7, color='blue', edgecolor='black')
        axes[1, 1].set_title('Signal Confidence Distribution')
        axes[1, 1].set_xlabel('Confidence Level')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_file = f'{OUTPUT_DIR}/kaggle_analysis_{timestamp}.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"📈 Visualization saved: {plot_file}")
        
        # 10. Display summary
        print("\n" + "=" * 60)
        print("🎉 KAGGLE STRATEGY RESULTS SUMMARY")
        print("=" * 60)
        
        # Extract key metrics
        accuracy_metrics = evaluation_results.get('accuracy_metrics', {})
        portfolio_performance = evaluation_results.get('portfolio_performance', {})
        signal_quality = evaluation_results.get('signal_quality', {})
        
        print(f"📊 MODEL PERFORMANCE:")
        print(f"   Directional Accuracy: {accuracy_metrics.get('directional_accuracy', 'N/A')}%")
        print(f"   R² Score: {accuracy_metrics.get('r2_score', 'N/A')}")
        print(f"   RMSE: {accuracy_metrics.get('rmse', 'N/A')}")
        
        print(f"\n💰 PORTFOLIO PERFORMANCE:")
        print(f"   Total Return: {portfolio_performance.get('total_return_pct', 'N/A')}%")
        print(f"   Final Value: ${portfolio_performance.get('final_value', 'N/A'):,.2f}")
        print(f"   Sharpe Ratio: {portfolio_performance.get('sharpe_ratio', 'N/A')}")
        print(f"   Max Drawdown: {portfolio_performance.get('max_drawdown_pct', 'N/A')}%")
        
        print(f"\n🎯 SIGNAL QUALITY:")
        print(f"   Total Signals: {signal_quality.get('total_signals', 'N/A')}")
        print(f"   Buy Signals: {signal_quality.get('buy_signals', 'N/A')}")
        print(f"   Sell Signals: {signal_quality.get('sell_signals', 'N/A')}")
        print(f"   Average Confidence: {signal_quality.get('average_confidence', 'N/A')}")
        print(f"   High Confidence Signals: {signal_quality.get('high_confidence_signals', 'N/A')}")
        
        print(f"\n✅ SUCCESS! All results saved to /kaggle/working/")
        print(f"📁 Generated files:")
        print(f"   • {signals_file}")
        print(f"   • {report_file}")
        print(f"   • {plot_file}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

# ============================================================================
# EXECUTE THE STRATEGY
# ============================================================================

if __name__ == "__main__":
    print("🚀 STARTING KAGGLE LSTM TRADING STRATEGY")
    print("=" * 70)
    print(f"📍 Data Path: {DATA_PATH}")
    print(f"📁 Output Directory: {OUTPUT_DIR}")
    print(f"🎯 Using {len(FOCUSED_INDICATORS)} focused indicators")
    print("=" * 70)
    
    # Run the strategy
    success = run_kaggle_focused_lstm()
    
    if success:
        print(f"\n🎉 STRATEGY COMPLETED SUCCESSFULLY!")
        print(f"📈 Check the visualizations above")
        print(f"📁 Download CSV files from /kaggle/working/")
        print(f"💡 Tip: Modify LSTM_CONFIG above to experiment with different settings")
    else:
        print(f"\n❌ STRATEGY FAILED!")
        print(f"💡 Check the error messages above and verify your data path")
        print(f"🔧 Ensure your CSV contains the required columns and indicators")
