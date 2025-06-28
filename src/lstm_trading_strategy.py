"""
LSTM Trading Signal Strategy with Performance Evaluation

This script creates LSTM predictions for stock price movements and evaluates
trading signals with realistic gain/loss calculations, stop losses, and accuracy metrics.
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Deep Learning Libraries
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.decomposition import PCA
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    from scipy import stats
    print("✅ TensorFlow, scikit-learn, and scipy imported successfully")
except ImportError as e:
    print(f"❌ Missing required libraries: {e}")
    print("Please install: pip install tensorflow scikit-learn scipy")
    exit(1)

import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os

class LSTMTradingStrategy:
    def __init__(self, 
                 sequence_length=60,
                 target_horizon=5,
                 target_gain=0.10,
                 stop_loss=-0.03,
                 test_split=0.2,
                 use_advanced_preprocessing=True,
                 outlier_threshold=1.5,
                 pca_variance_threshold=0.95):
        """
        Initialize LSTM Trading Strategy
        
        Parameters:
        sequence_length: Number of days to look back for LSTM input
        target_horizon: Number of days to hold position
        target_gain: Target profit percentage (10% = 0.10)
        stop_loss: Stop loss percentage (-3% = -0.03)
        test_split: Percentage of data for testing
        use_advanced_preprocessing: Enable advanced preprocessing (interpolation, outlier removal, PCA)
        outlier_threshold: IQR multiplier for outlier detection (1.5 = standard)
        pca_variance_threshold: Minimum variance to retain in PCA (0.95 = 95%)
        """
        self.sequence_length = sequence_length
        self.target_horizon = target_horizon
        self.target_gain = target_gain
        self.stop_loss = stop_loss
        self.test_split = test_split
        self.use_advanced_preprocessing = use_advanced_preprocessing
        self.outlier_threshold = outlier_threshold
        self.pca_variance_threshold = pca_variance_threshold
        
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.pca = None
        self.model = None
        self.history = None
        
    def prepare_data(self, df, features=['close']):
        """
        Prepare data for LSTM training with advanced preprocessing options
        """
        print("🚀 Preparing data for LSTM with advanced preprocessing...")
        
        # Ensure we have the required columns
        feature_cols = []
        for feature in features:
            if feature.lower() in [col.lower() for col in df.columns]:
                # Find the actual column name (case insensitive)
                actual_col = next(col for col in df.columns if col.lower() == feature.lower())
                feature_cols.append(actual_col)
            else:
                print(f"Warning: Feature '{feature}' not found in data")
        
        if not feature_cols:
            raise ValueError("No valid features found in the data")
        
        # Sort by date if date column exists
        if 'date' in [col.lower() for col in df.columns]:
            date_col = next(col for col in df.columns if col.lower() == 'date')
            df = df.sort_values(date_col)
        
        # Extract features into DataFrame for advanced preprocessing
        data_df = df[feature_cols].copy()
        print(f"📊 Initial data shape: {data_df.shape}")
        
        if self.use_advanced_preprocessing:
            print("\n🔧 ADVANCED PREPROCESSING PIPELINE:")
            
            # Step 1: Handle missing values with linear interpolation
            data_df = self.handle_missing_values_advanced(data_df)
            
            # Step 2: Remove outliers using IQR-based filtering
            data_df = self.remove_outliers_iqr(data_df)
            
            print(f"📊 Data shape after outlier removal: {data_df.shape}")
            
            # Ensure we still have enough data
            if len(data_df) < self.sequence_length * 5:
                print("⚠️  Warning: Limited data after outlier removal. Consider adjusting outlier_threshold.")
            
            # Convert back to numpy array for scaling
            data = data_df.values
            
            # Step 3: Normalize features via Min-Max scaling
            print("🔧 Applying Min-Max scaling...")
            scaled_data = self.scaler.fit_transform(data)
            
            # Step 4: Apply PCA for dimensionality reduction (retain 95% variance)
            if len(feature_cols) > 2:  # Only apply PCA if we have multiple features
                scaled_data = self.apply_pca_reduction(scaled_data)
            
        else:
            print("\n📝 BASIC PREPROCESSING:")
            # Basic preprocessing (original approach)
            data = data_df.values
            
            # Handle missing values (basic approach)
            if np.isnan(data).any():
                print("🔧 Handling missing values with forward/backward fill...")
                data_df = pd.DataFrame(data).fillna(method='ffill').fillna(method='bfill')
                data = data_df.values
            
            # Scale the data
            print("🔧 Applying Min-Max scaling...")
            scaled_data = self.scaler.fit_transform(data)
        
        print(f"📊 Final preprocessed data shape: {scaled_data.shape}")
        
        # Create sequences for LSTM
        print("🔧 Creating LSTM sequences...")
        X, y = [], []
        for i in range(self.sequence_length, len(scaled_data)):
            X.append(scaled_data[i-self.sequence_length:i])
            y.append(scaled_data[i, 0])  # Predict the first feature (usually close price)
        
        X, y = np.array(X), np.array(y)
        
        # Split into train and test
        split_idx = int(len(X) * (1 - self.test_split))
        
        self.X_train = X[:split_idx]
        self.X_test = X[split_idx:]
        self.y_train = y[:split_idx]
        self.y_test = y[split_idx:]
        
        # Store original data for evaluation
        self.original_data = df[feature_cols[0]].values  # Close prices
        self.test_start_idx = len(scaled_data) - len(self.y_test)
        
        print(f"✅ Data preparation complete:")
        print(f"   Training samples: {len(self.X_train)}")
        print(f"   Test samples: {len(self.X_test)}")
        print(f"   Feature dimensions: {self.X_train.shape[2]}")
        print(f"   Advanced preprocessing: {'Enabled' if self.use_advanced_preprocessing else 'Disabled'}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def build_model(self, lstm_units=[50, 50], dropout_rate=0.2, optimize_features=False):
        """
        Build LSTM model with optional feature optimization (disabled by default for better accuracy)
        """
        print("Building LSTM model...")
        
        # Store feature names for optimization
        if hasattr(self, 'X_train'):
            self.feature_columns = [f'Feature_{i}' for i in range(self.X_train.shape[2])]
        
        # Optimize feature selection if requested (conservative approach)
        if optimize_features and hasattr(self, 'X_train') and self.X_train.shape[2] > 10:
            print("🔍 Optimizing feature selection (conservative mode)...")
            # Flatten for feature selection
            X_train_2d = self.X_train.reshape(self.X_train.shape[0], -1)
            y_train_1d = self.y_train.flatten()
            
            # Use conservative feature optimization (keep more features)
            max_features = max(self.X_train.shape[2] // 2, 8)  # Keep at least half or 8 features
            X_train_optimized = self.optimize_feature_selection(X_train_2d, y_train_1d, max_features=max_features)
            
            # Update training data
            self.X_train = X_train_optimized.reshape(self.X_train.shape[0], self.X_train.shape[1], -1)
            self.X_test = self.X_test[:, :, self.selected_features]
            
            print(f"✅ Features optimized conservatively: {self.X_train.shape[2]} selected features")
        elif optimize_features:
            print("🔍 Skipping feature optimization: Too few features or not enough data")
        
        self.model = Sequential()
        
        # First LSTM layer
        self.model.add(LSTM(units=lstm_units[0], 
                           return_sequences=True if len(lstm_units) > 1 else False,
                           input_shape=(self.X_train.shape[1], self.X_train.shape[2])))
        self.model.add(Dropout(dropout_rate))
        
        # Additional LSTM layers
        for i, units in enumerate(lstm_units[1:], 1):
            return_seq = i < len(lstm_units) - 1
            self.model.add(LSTM(units=units, return_sequences=return_seq))
            self.model.add(Dropout(dropout_rate))
        
        # Output layer
        self.model.add(Dense(1))
        
        # Compile model with standard learning rate for stability
        self.model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
        
        print("Model architecture:")
        self.model.summary()
        
    def train_model(self, epochs=50, batch_size=32, validation_split=0.1, verbose=1):
        """
        Train the LSTM model
        """
        print("Training LSTM model...")
        
        # Callbacks for better training
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
        ]
        
        self.history = self.model.fit(
            self.X_train, self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=verbose
        )
        
        print("Model training completed!")
        
    def inverse_transform_price(self, scaled_values):
        """
        Helper function to properly inverse transform price values
        Handles both regular scaling and PCA transformation
        """
        if np.isscalar(scaled_values):
            scaled_values = [scaled_values]
        
        # If PCA was applied, we need to handle the transformation differently
        if self.pca is not None:
            # For PCA case, we can only approximate the inverse transform
            # since we're dealing with reduced dimensionality
            
            # Create dummy array with PCA components
            dummy_pca = np.zeros((len(scaled_values), self.pca.n_components_))
            dummy_pca[:, 0] = scaled_values  # First component represents the price
            
            # Inverse PCA transform to get back to original feature space
            dummy_original = self.pca.inverse_transform(dummy_pca)
            
            # Now inverse scale using the first feature (price)
            dummy_for_scaler = np.zeros((len(scaled_values), self.scaler.n_features_in_))
            dummy_for_scaler[:, 0] = dummy_original[:, 0]
            
            return self.scaler.inverse_transform(dummy_for_scaler)[:, 0]
        else:
            # Original approach for non-PCA case
            dummy = np.zeros((len(scaled_values), self.scaler.n_features_in_))
            dummy[:, 0] = scaled_values  # First feature is always the close price
            
            # Inverse transform and return only the price column
            return self.scaler.inverse_transform(dummy)[:, 0]

    def make_predictions(self):
        """
        Make predictions on test data
        """
        print("Making predictions...")
        
        # Predict on test data
        y_pred_scaled = self.model.predict(self.X_test)
        
        # Inverse transform predictions using helper function
        y_pred = self.inverse_transform_price(y_pred_scaled.flatten())
        
        # Inverse transform actual values using helper function
        y_actual = self.inverse_transform_price(self.y_test)
        
        self.y_pred = y_pred
        self.y_actual = y_actual
        
        # Calculate basic metrics
        mse = mean_squared_error(y_actual, y_pred)
        mae = mean_absolute_error(y_actual, y_pred)
        rmse = np.sqrt(mse)
        
        print(f"Prediction Metrics:")
        print(f"  MSE: {mse:.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        
        return y_pred, y_actual
    
    def generate_trading_signals(self, df):
        """
        Generate comprehensive trading signals and evaluate performance with detailed metrics
        """
        print("Generating comprehensive trading signals...")
        
        # Get the actual close prices for the test period
        close_col = next(col for col in df.columns if col.lower() == 'close')
        all_close_prices = df[close_col].values
        
        # Get close prices for test period
        test_close_prices = all_close_prices[self.test_start_idx + self.sequence_length:]
        
        # Debug information
        print(f"Debug info:")
        print(f"  y_pred length: {len(self.y_pred)}")
        print(f"  test_close_prices length: {len(test_close_prices)}")
        print(f"  target_horizon: {self.target_horizon}")
        
        # Safety check: ensure we have enough data to generate signals
        if len(self.y_pred) < self.target_horizon or len(test_close_prices) < self.target_horizon:
            print("⚠️ Warning: Insufficient data for signal generation")
            print(f"   Need at least {self.target_horizon} data points")
            print(f"   Available predictions: {len(self.y_pred)}")
            print(f"   Available test prices: {len(test_close_prices)}")
            return []
        
        # Ensure we have date information
        if hasattr(df.index, 'strftime'):
            # Index is datetime
            test_dates = df.index[self.test_start_idx + self.sequence_length:]
        elif 'date' in [col.lower() for col in df.columns]:
            # Date is a column
            date_col = next(col for col in df.columns if col.lower() == 'date')
            test_dates = df[date_col].iloc[self.test_start_idx + self.sequence_length:]
        else:
            # Create synthetic dates
            test_dates = pd.date_range(start='2023-01-01', periods=len(test_close_prices), freq='D')
        
        results = []
        portfolio_value = 100000  # Starting portfolio value
        current_portfolio = portfolio_value
        
        # Generate signals for each prediction
        # Ensure we don't go out of bounds for both predictions and actual prices
        max_signals = min(len(self.y_pred), len(test_close_prices)) - self.target_horizon
        max_signals = max(0, max_signals)  # Ensure it's not negative
        
        print(f"Generating signals for {max_signals} trading opportunities...")
        
        for i in range(max_signals):
            # Current and future predictions (scaled back to original prices)
            current_pred_scaled = self.inverse_transform_price([self.y_pred[i]])[0]
            if i + 1 < len(self.y_pred):
                future_pred_scaled = self.inverse_transform_price([self.y_pred[i + 1]])[0]
            else:
                future_pred_scaled = current_pred_scaled
            
            # Current and target prices - ensure we don't go out of bounds
            start_price = test_close_prices[i]
            target_price = test_close_prices[i + self.target_horizon] if i + self.target_horizon < len(test_close_prices) else start_price
            
            # Calculate prediction confidence (simplified for better accuracy)
            pred_change = (future_pred_scaled - current_pred_scaled) / current_pred_scaled
            confidence = min(abs(pred_change) * 100, 100)  # Simple percentage-based confidence
            
            # Determine signal based on prediction direction and confidence  
            signal_strength = "Strong" if confidence > 2.0 else "Medium" if confidence > 1.0 else "Weak"
            predicted_trend = 'BUY' if future_pred_scaled > current_pred_scaled else 'SELL'
            actual_trend = 'BUY' if target_price > start_price else 'SELL'
            
            # Calculate detailed performance metrics
            pct_change = (target_price - start_price) / start_price
            gain_loss = pct_change * 100
            correct_prediction = predicted_trend == actual_trend
            
            # Simulate realistic trading with intraday monitoring
            exit_price = target_price
            exit_reason = "Target_Duration"
            actual_duration = self.target_horizon
            max_gain_during_period = 0
            max_loss_during_period = 0
            
            # Check intraday performance
            for day in range(1, min(self.target_horizon + 1, len(test_close_prices) - i)):
                if i + day < len(test_close_prices):
                    daily_price = test_close_prices[i + day]
                    daily_return = (daily_price - start_price) / start_price
                    
                    # Track max gain/loss
                    max_gain_during_period = max(max_gain_during_period, daily_return * 100)
                    max_loss_during_period = min(max_loss_during_period, daily_return * 100)
                    
                    # Check exit conditions
                    if daily_return <= self.stop_loss:
                        exit_price = daily_price
                        exit_reason = "Stop_Loss"
                        actual_duration = day
                        gain_loss = daily_return * 100
                        break
                    elif daily_return >= self.target_gain:
                        exit_price = daily_price
                        exit_reason = "Target_Hit"
                        actual_duration = day
                        gain_loss = daily_return * 100
                        break
            
            # Calculate risk metrics
            volatility = abs(max_gain_during_period - max_loss_during_period)
            risk_reward_ratio = abs(self.target_gain * 100) / abs(self.stop_loss * 100) if self.stop_loss != 0 else 0
            
            # Update portfolio tracking
            trade_value = current_portfolio * 0.1  # Use 10% of portfolio per trade
            if predicted_trend == 'BUY':
                shares = trade_value / start_price
                trade_profit = shares * (exit_price - start_price)
                current_portfolio += trade_profit
            
            # Calculate additional metrics
            holding_period_return = gain_loss
            annualized_return = (holding_period_return / actual_duration) * 365 if actual_duration > 0 else 0
            
            # Format date
            try:
                if i < len(test_dates):
                    if hasattr(test_dates.iloc[i], 'strftime'):
                        date_str = test_dates.iloc[i].strftime("%Y-%m-%d")
                    else:
                        date_str = str(test_dates.iloc[i])[:10]
                else:
                    date_str = f"Day_{i}"
            except:
                date_str = f"Day_{i}"
            
            # Create comprehensive signal record
            signal_record = {
                # Basic Information
                'Signal_ID': f"SIG_{i+1:04d}",
                'Date': date_str,
                'Signal_Type': predicted_trend,
                'Signal_Strength': signal_strength,
                'Confidence_Score': round(confidence, 2),
                
                # Price Information
                'Entry_Price': round(start_price, 2),
                'Exit_Price': round(exit_price, 2),
                'Predicted_Price': round(future_pred_scaled, 2),
                'Target_Price': round(start_price * (1 + self.target_gain), 2),
                'Stop_Loss_Price': round(start_price * (1 + self.stop_loss), 2),
                
                # Performance Metrics
                'Actual_Gain_Loss_Pct': round(gain_loss, 2),
                'Actual_Gain_Loss_Amount': round(trade_value * (gain_loss / 100), 2) if predicted_trend == 'BUY' else 0,
                'Max_Gain_During_Period': round(max_gain_during_period, 2),
                'Max_Loss_During_Period': round(max_loss_during_period, 2),
                'Volatility_During_Period': round(volatility, 2),
                
                # Signal Evaluation
                'Prediction_Accuracy': 'Correct' if correct_prediction else 'Incorrect',
                'Target_Hit': 'Yes' if exit_reason == "Target_Hit" else 'No',
                'Stop_Loss_Hit': 'Yes' if exit_reason == "Stop_Loss" else 'No',
                'Exit_Reason': exit_reason,
                
                # Duration and Timing
                'Planned_Duration_Days': self.target_horizon,
                'Actual_Duration_Days': actual_duration,
                'Annualized_Return_Pct': round(annualized_return, 2),
                
                # Risk Metrics
                'Risk_Reward_Ratio': round(risk_reward_ratio, 2),
                'Portfolio_Value_After_Trade': round(current_portfolio, 2),
                
                # Trends
                'Predicted_Trend': 'Bullish' if predicted_trend == 'BUY' else 'Bearish',
                'Actual_Trend': 'Bullish' if actual_trend == 'BUY' else 'Bearish',
                
                # Additional Analysis
                'Trade_Success': 'Profitable' if gain_loss > 0 else 'Loss',
                'High_Confidence': 'Yes' if confidence > 2.0 else 'No'
            }
            
            results.append(signal_record)
        
        self.results_df = pd.DataFrame(results)
        print(f"Generated {len(results)} trading signals with comprehensive evaluation")
        return self.results_df
    
    def calculate_summary_statistics(self):
        """
        Calculate comprehensive summary statistics
        """
        if not hasattr(self, 'results_df'):
            raise ValueError("Must generate trading signals first")
        
        df = self.results_df
        
        # Basic statistics
        total_signals = len(df)
        correct_predictions = (df['Prediction_Accuracy'] == 'Correct').sum()
        accuracy = (correct_predictions / total_signals) * 100 if total_signals > 0 else 0
        
        # Performance statistics
        bullish_signals = df[df['Signal_Type'] == 'BUY']
        bearish_signals = df[df['Signal_Type'] == 'SELL']
        
        correct_bullish = (bullish_signals['Prediction_Accuracy'] == 'Correct').sum()
        correct_bearish = (bearish_signals['Prediction_Accuracy'] == 'Correct').sum()
        
        bullish_accuracy = (correct_bullish / len(bullish_signals)) * 100 if len(bullish_signals) > 0 else 0
        bearish_accuracy = (correct_bearish / len(bearish_signals)) * 100 if len(bearish_signals) > 0 else 0
        
        # Return statistics
        correct_mask = df['Prediction_Accuracy'] == 'Correct'
        avg_gain_correct = df[correct_mask]['Actual_Gain_Loss_Pct'].mean() if correct_predictions > 0 else 0
        avg_loss_incorrect = df[~correct_mask]['Actual_Gain_Loss_Pct'].mean() if (total_signals - correct_predictions) > 0 else 0
        total_return = df['Actual_Gain_Loss_Pct'].sum()
        
        # Risk metrics
        target_hits = (df['Target_Hit'] == 'Yes').sum()
        stop_losses = (df['Stop_Loss_Hit'] == 'Yes').sum()
        avg_duration = df['Actual_Duration_Days'].mean()
        
        # Volatility
        returns_std = df['Actual_Gain_Loss_Pct'].std()
        sharpe_ratio = (df['Actual_Gain_Loss_Pct'].mean() / returns_std) if returns_std > 0 else 0
        
        # Win/Loss ratios
        winning_trades = len(df[df['Actual_Gain_Loss_Pct'] > 0])
        losing_trades = len(df[df['Actual_Gain_Loss_Pct'] < 0])
        win_rate = (winning_trades / total_signals) * 100 if total_signals > 0 else 0
        
        avg_win = df[df['Actual_Gain_Loss_Pct'] > 0]['Actual_Gain_Loss_Pct'].mean() if winning_trades > 0 else 0
        avg_loss = df[df['Actual_Gain_Loss_Pct'] < 0]['Actual_Gain_Loss_Pct'].mean() if losing_trades > 0 else 0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        
        self.summary = {
            'Total_Signals': total_signals,
            'Correct_Predictions': correct_predictions,
            'Overall_Accuracy_Pct': round(accuracy, 2),
            'Bullish_Signals': len(bullish_signals),
            'Bearish_Signals': len(bearish_signals),
            'Bullish_Accuracy_Pct': round(bullish_accuracy, 2),
            'Bearish_Accuracy_Pct': round(bearish_accuracy, 2),
            'Avg_Gain_Correct_Pct': round(avg_gain_correct, 2),
            'Avg_Loss_Incorrect_Pct': round(avg_loss_incorrect, 2),
            'Total_Return_Pct': round(total_return, 2),
            'Target_Hits': target_hits,
            'Stop_Losses': stop_losses,
            'Avg_Duration_Days': round(avg_duration, 1),
            'Winning_Trades': winning_trades,
            'Losing_Trades': losing_trades,
            'Win_Rate_Pct': round(win_rate, 2),
            'Avg_Win_Pct': round(avg_win, 2),
            'Avg_Loss_Pct': round(avg_loss, 2),
            'Profit_Factor': round(profit_factor, 2),
            'Volatility_Pct': round(returns_std, 2),
            'Sharpe_Ratio': round(sharpe_ratio, 2)
        }
        
        return self.summary
    
    def plot_results(self, save_path="../output/"):
        """
        Create comprehensive plots of the results
        """
        if not hasattr(self, 'results_df'):
            raise ValueError("Must generate trading signals first")
        
        # Create output directory if it doesn't exist
        os.makedirs(save_path, exist_ok=True)
        
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('LSTM Trading Strategy Performance Analysis', fontsize=16, fontweight='bold')
        
        # 1. Predictions vs Actual
        axes[0, 0].plot(self.y_actual[:100], label='Actual', alpha=0.8)
        axes[0, 0].plot(self.y_pred[:100], label='Predicted', alpha=0.8)
        axes[0, 0].set_title('Price Predictions vs Actual (First 100 days)')
        axes[0, 0].set_xlabel('Days')
        axes[0, 0].set_ylabel('Price')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Accuracy by Signal Type
        accuracy_data = self.results_df.groupby('Signal_Type').apply(lambda x: (x['Prediction_Accuracy'] == 'Correct').mean() * 100)
        accuracy_data.plot(kind='bar', ax=axes[0, 1], color=['red', 'green'])
        axes[0, 1].set_title('Accuracy by Signal Type')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].set_xticklabels(accuracy_data.index, rotation=0)
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Return Distribution
        axes[0, 2].hist(self.results_df['Actual_Gain_Loss_Pct'], bins=30, alpha=0.7, color='blue', edgecolor='black')
        axes[0, 2].axvline(0, color='red', linestyle='--', label='Break-even')
        axes[0, 2].axvline(self.target_gain * 100, color='green', linestyle='--', label='Target')
        axes[0, 2].axvline(self.stop_loss * 100, color='red', linestyle='--', label='Stop Loss')
        axes[0, 2].set_title('Distribution of Returns')
        axes[0, 2].set_xlabel('Return (%)')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Cumulative Returns
        cumulative_returns = self.results_df['Actual_Gain_Loss_Pct'].cumsum()
        axes[1, 0].plot(cumulative_returns.index, cumulative_returns.values, color='purple')
        axes[1, 0].set_title('Cumulative Returns')
        axes[1, 0].set_xlabel('Trade Number')
        axes[1, 0].set_ylabel('Cumulative Return (%)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Training History
        if self.history:
            axes[1, 1].plot(self.history.history['loss'], label='Training Loss')
            if 'val_loss' in self.history.history:
                axes[1, 1].plot(self.history.history['val_loss'], label='Validation Loss')
            axes[1, 1].set_title('Model Training History')
            axes[1, 1].set_xlabel('Epochs')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Win/Loss Analysis
        win_loss_data = pd.DataFrame({
            'Wins': [len(self.results_df[self.results_df['Actual_Gain_Loss_Pct'] > 0])],
            'Losses': [len(self.results_df[self.results_df['Actual_Gain_Loss_Pct'] < 0])]
        })
        win_loss_data.T.plot(kind='bar', ax=axes[1, 2], color=['green', 'red'])
        axes[1, 2].set_title('Win/Loss Distribution')
        axes[1, 2].set_ylabel('Number of Trades')
        axes[1, 2].set_xticklabels(['Wins', 'Losses'], rotation=0)
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'lstm_strategy_analysis.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Plots saved to {save_path}")
    
    def save_results(self, save_path="../output/"):
        """
        Save all results to files
        """
        os.makedirs(save_path, exist_ok=True)
        
        # Save detailed results
        results_file = os.path.join(save_path, 'lstm_trade_signals.csv')
        self.results_df.to_csv(results_file, index=False)
        
        # Save summary statistics
        summary_file = os.path.join(save_path, 'lstm_strategy_summary.csv')
        summary_df = pd.DataFrame([self.summary])
        summary_df.to_csv(summary_file, index=False)
        
        # Save detailed summary report
        report_file = os.path.join(save_path, 'lstm_strategy_report.txt')
        with open(report_file, 'w') as f:
            f.write("LSTM TRADING STRATEGY PERFORMANCE REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Strategy Parameters:\n")
            f.write(f"  - Sequence Length: {self.sequence_length} days\n")
            f.write(f"  - Target Horizon: {self.target_horizon} days\n")
            f.write(f"  - Target Gain: {self.target_gain * 100}%\n")
            f.write(f"  - Stop Loss: {self.stop_loss * 100}%\n\n")
            
            f.write("Performance Summary:\n")
            for key, value in self.summary.items():
                f.write(f"  - {key.replace('_', ' ')}: {value}\n")
            
            f.write(f"\nTop 10 Best Performing Signals:\n")
            top_signals = self.results_df.nlargest(10, 'Actual_Gain_Loss_Pct')[['Date', 'Signal_Type', 'Actual_Gain_Loss_Pct', 'Prediction_Accuracy']]
            f.write(top_signals.to_string(index=False))
            
            f.write(f"\n\nTop 10 Worst Performing Signals:\n")
            worst_signals = self.results_df.nsmallest(10, 'Actual_Gain_Loss_Pct')[['Date', 'Signal_Type', 'Actual_Gain_Loss_Pct', 'Prediction_Accuracy']]
            f.write(worst_signals.to_string(index=False))
        
        print(f"✅ Results saved to {save_path}")
        print(f"   - Detailed signals: {results_file}")
        print(f"   - Summary: {summary_file}")
        print(f"   - Report: {report_file}")

    def generate_enhanced_signal_evaluation(self):
        """
        Generate comprehensive signal evaluation with detailed trading metrics
        """
        if not hasattr(self, 'results_df'):
            raise ValueError("Must generate trading signals first")
        
        print("Generating enhanced signal evaluation...")
        
        # Create enhanced evaluation DataFrame with additional metrics
        enhanced_signals = []
        
        for idx, row in self.results_df.iterrows():
            # Calculate additional metrics for each signal
            trade_efficiency = (row['Actual_Duration_Days'] / self.target_horizon) * 100
            risk_adjusted_return = row['Actual_Gain_Loss_Pct'] / row['Volatility_During_Period'] if row['Volatility_During_Period'] > 0 else 0
            
            # Determine trade grade based on performance
            if row['Actual_Gain_Loss_Pct'] >= self.target_gain * 100:
                trade_grade = 'A'  # Excellent
            elif row['Actual_Gain_Loss_Pct'] > 0:
                trade_grade = 'B'  # Good
            elif row['Actual_Gain_Loss_Pct'] > self.stop_loss * 100:
                trade_grade = 'C'  # Acceptable
            else:
                trade_grade = 'D'  # Poor
            
            enhanced_signal = {
                **row.to_dict(),  # Include all original columns
                'Trade_Efficiency_Pct': round(trade_efficiency, 2),
                'Risk_Adjusted_Return': round(risk_adjusted_return, 2),
                'Trade_Grade': trade_grade,
                'Signal_Quality_Score': self._calculate_signal_quality_score(row)
            }
            
            enhanced_signals.append(enhanced_signal)
        
        self.enhanced_results_df = pd.DataFrame(enhanced_signals)
        return self.enhanced_results_df
    
    def _calculate_signal_quality_score(self, signal_row):
        """
        Calculate a quality score for each signal (0-100)
        """
        score = 0
        
        # Confidence score (0-30 points)
        score += min(signal_row['Confidence_Score'] * 3, 30)
        
        # Accuracy (0-25 points)
        if signal_row['Prediction_Accuracy'] == 'Correct':
            score += 25
        
        # Profitability (0-25 points)
        if signal_row['Actual_Gain_Loss_Pct'] > 0:
            score += min(signal_row['Actual_Gain_Loss_Pct'] * 2.5, 25)
        
        # Risk management (0-20 points)
        if signal_row['Stop_Loss_Hit'] == 'No':
            score += 10
        if signal_row['Target_Hit'] == 'Yes':
            score += 10
        
        return round(min(score, 100), 1)
    
    def create_signal_evaluation_report(self, save_path="../output/"):
        """
        Create a comprehensive signal evaluation report
        """
        if not hasattr(self, 'enhanced_results_df'):
            self.generate_enhanced_signal_evaluation()
        
        os.makedirs(save_path, exist_ok=True)
        
        report_path = os.path.join(save_path, 'lstm_signal_evaluation_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("LSTM TRADING SIGNAL EVALUATION REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            # Executive Summary
            f.write("EXECUTIVE SUMMARY\n")
            f.write("-" * 50 + "\n")
            f.write(f"Total Signals Generated: {len(self.enhanced_results_df)}\n")
            f.write(f"Overall Accuracy: {self.summary['Overall_Accuracy_Pct']}%\n")
            f.write(f"Win Rate: {self.summary['Win_Rate_Pct']}%\n")
            f.write(f"Total Return: {self.summary['Total_Return_Pct']}%\n")
            f.write(f"Sharpe Ratio: {self.summary['Sharpe_Ratio']}\n")
            f.write(f"Average Signal Quality Score: {self.enhanced_results_df['Signal_Quality_Score'].mean():.1f}/100\n\n")
            
            # Signal Type Analysis
            f.write("SIGNAL TYPE ANALYSIS\n")
            f.write("-" * 50 + "\n")
            buy_signals = self.enhanced_results_df[self.enhanced_results_df['Signal_Type'] == 'BUY']
            sell_signals = self.enhanced_results_df[self.enhanced_results_df['Signal_Type'] == 'SELL']
            
            f.write(f"BUY Signals: {len(buy_signals)} ({len(buy_signals)/len(self.enhanced_results_df)*100:.1f}%)\n")
            f.write(f"  - Accuracy: {(buy_signals['Prediction_Accuracy'] == 'Correct').mean()*100:.1f}%\n")
            f.write(f"  - Average Return: {buy_signals['Actual_Gain_Loss_Pct'].mean():.2f}%\n")
            f.write(f"  - Win Rate: {(buy_signals['Actual_Gain_Loss_Pct'] > 0).mean()*100:.1f}%\n\n")
            
            f.write(f"SELL Signals: {len(sell_signals)} ({len(sell_signals)/len(self.enhanced_results_df)*100:.1f}%)\n")
            f.write(f"  - Accuracy: {(sell_signals['Prediction_Accuracy'] == 'Correct').mean()*100:.1f}%\n")
            f.write(f"  - Average Return: {sell_signals['Actual_Gain_Loss_Pct'].mean():.2f}%\n")
            f.write(f"  - Win Rate: {(sell_signals['Actual_Gain_Loss_Pct'] > 0).mean()*100:.1f}%\n\n")
            
            # Risk Management Analysis
            f.write("RISK MANAGEMENT ANALYSIS\n")
            f.write("-" * 50 + "\n")
            target_hits = (self.enhanced_results_df['Target_Hit'] == 'Yes').sum()
            stop_losses = (self.enhanced_results_df['Stop_Loss_Hit'] == 'Yes').sum()
            f.write(f"Target Hits: {target_hits} ({target_hits/len(self.enhanced_results_df)*100:.1f}%)\n")
            f.write(f"Stop Losses: {stop_losses} ({stop_losses/len(self.enhanced_results_df)*100:.1f}%)\n")
            f.write(f"Average Holding Period: {self.enhanced_results_df['Actual_Duration_Days'].mean():.1f} days\n")
            f.write(f"Maximum Gain: {self.enhanced_results_df['Actual_Gain_Loss_Pct'].max():.2f}%\n")
            f.write(f"Maximum Loss: {self.enhanced_results_df['Actual_Gain_Loss_Pct'].min():.2f}%\n\n")
            
            # Top Performing Signals
            f.write("TOP 10 PERFORMING SIGNALS\n")
            f.write("-" * 50 + "\n")
            top_signals = self.enhanced_results_df.nlargest(10, 'Actual_Gain_Loss_Pct')
            for idx, signal in top_signals.iterrows():
                f.write(f"{signal['Date']}: {signal['Signal_Type']} | ")
                f.write(f"Return: {signal['Actual_Gain_Loss_Pct']:.2f}% | ")
                f.write(f"Grade: {signal['Trade_Grade']} | ")
                f.write(f"Quality: {signal['Signal_Quality_Score']}/100\n")
            
            f.write("\n" + "WORST 10 PERFORMING SIGNALS\n")
            f.write("-" * 50 + "\n")
            worst_signals = self.enhanced_results_df.nsmallest(10, 'Actual_Gain_Loss_Pct')
            for idx, signal in worst_signals.iterrows():
                f.write(f"{signal['Date']}: {signal['Signal_Type']} | ")
                f.write(f"Return: {signal['Actual_Gain_Loss_Pct']:.2f}% | ")
                f.write(f"Grade: {signal['Trade_Grade']} | ")
                f.write(f"Quality: {signal['Signal_Quality_Score']}/100\n")
            
            # Trade Grade Distribution
            f.write("\n" + "TRADE GRADE DISTRIBUTION\n")
            f.write("-" * 50 + "\n")
            grade_counts = self.enhanced_results_df['Trade_Grade'].value_counts().sort_index()
            for grade, count in grade_counts.items():
                f.write(f"Grade {grade}: {count} trades ({count/len(self.enhanced_results_df)*100:.1f}%)\n")
            
            # Confidence Analysis
            f.write("\n" + "CONFIDENCE ANALYSIS\n")
            f.write("-" * 50 + "\n")
            high_conf = self.enhanced_results_df[self.enhanced_results_df['High_Confidence'] == 'Yes']
            low_conf = self.enhanced_results_df[self.enhanced_results_df['High_Confidence'] == 'No']
            
            f.write(f"High Confidence Signals: {len(high_conf)} ({len(high_conf)/len(self.enhanced_results_df)*100:.1f}%)\n")
            f.write(f"  - Accuracy: {(high_conf['Prediction_Accuracy'] == 'Correct').mean()*100:.1f}%\n")
            f.write(f"  - Average Return: {high_conf['Actual_Gain_Loss_Pct'].mean():.2f}%\n")
            
            f.write(f"\nLow Confidence Signals: {len(low_conf)} ({len(low_conf)/len(self.enhanced_results_df)*100:.1f}%)\n")
            f.write(f"  - Accuracy: {(low_conf['Prediction_Accuracy'] == 'Correct').mean()*100:.1f}%\n")
            f.write(f"  - Average Return: {low_conf['Actual_Gain_Loss_Pct'].mean():.2f}%\n")
            
            # Model Performance
            f.write("\n" + "MODEL PERFORMANCE METRICS\n")
            f.write("-" * 50 + "\n")
            f.write(f"RMSE: {self.summary.get('RMSE', 'N/A')}\n")
            f.write(f"MAE: {self.summary.get('MAE', 'N/A')}\n")
            f.write(f"Profit Factor: {self.summary['Profit_Factor']}\n")
            f.write(f"Volatility: {self.summary['Volatility_Pct']}%\n")
        
        print(f"✅ Enhanced signal evaluation report saved to: {report_path}")
        return report_path

    def save_enhanced_results(self, save_path="../output/"):
        """
        Save enhanced results with comprehensive signal evaluation
        """
        if not hasattr(self, 'enhanced_results_df'):
            self.generate_enhanced_signal_evaluation()
        
        os.makedirs(save_path, exist_ok=True)
        
        # Save enhanced signals CSV
        enhanced_file = os.path.join(save_path, 'lstm_enhanced_trade_signals.csv')
        self.enhanced_results_df.to_csv(enhanced_file, index=False)
        
        # Create evaluation report
        report_path = self.create_signal_evaluation_report(save_path)
        
        # Save summary with additional metrics
        summary_file = os.path.join(save_path, 'lstm_enhanced_strategy_summary.csv')
        enhanced_summary = {
            **self.summary,
            'Average_Signal_Quality_Score': round(self.enhanced_results_df['Signal_Quality_Score'].mean(), 1),
            'Grade_A_Trades': len(self.enhanced_results_df[self.enhanced_results_df['Trade_Grade'] == 'A']),
            'Grade_B_Trades': len(self.enhanced_results_df[self.enhanced_results_df['Trade_Grade'] == 'B']),
            'Grade_C_Trades': len(self.enhanced_results_df[self.enhanced_results_df['Trade_Grade'] == 'C']),
            'Grade_D_Trades': len(self.enhanced_results_df[self.enhanced_results_df['Trade_Grade'] == 'D']),
            'High_Quality_Signals_Pct': round((self.enhanced_results_df['Signal_Quality_Score'] >= 70).mean() * 100, 1)
        }
        
        pd.DataFrame([enhanced_summary]).to_csv(summary_file, index=False)
        
        print(f"✅ Enhanced results saved:")
        print(f"   - Enhanced signals: {enhanced_file}")
        print(f"   - Evaluation report: {report_path}")
        print(f"   - Enhanced summary: {summary_file}")
        
        return enhanced_file, report_path, summary_file

    def handle_missing_values_advanced(self, data_df):
        """
        Handle missing values using linear interpolation
        """
        print("🔧 Applying linear interpolation for missing values...")
        
        # Count missing values before
        missing_before = data_df.isnull().sum().sum()
        
        if missing_before > 0:
            print(f"   Found {missing_before} missing values")
            
            # Apply linear interpolation
            data_df = data_df.interpolate(method='linear', limit_direction='both')
            
            # Fill any remaining NaN values at edges with forward/backward fill
            data_df = data_df.fillna(method='ffill').fillna(method='bfill')
            
            # Count missing values after
            missing_after = data_df.isnull().sum().sum()
            print(f"   Missing values after interpolation: {missing_after}")
        else:
            print("   No missing values found")
            
        return data_df
    
    def remove_outliers_iqr(self, data_df):
        """
        Remove outliers using IQR-based filtering
        """
        print(f"🔧 Removing outliers using IQR method (threshold: {self.outlier_threshold})...")
        
        rows_before = len(data_df)
        outlier_counts = {}
        
        for column in data_df.columns:
            Q1 = data_df[column].quantile(0.25)
            Q3 = data_df[column].quantile(0.75)
            IQR = Q3 - Q1
            
            # Define outlier bounds
            lower_bound = Q1 - self.outlier_threshold * IQR
            upper_bound = Q3 + self.outlier_threshold * IQR
            
            # Count outliers for this column
            outliers = ((data_df[column] < lower_bound) | (data_df[column] > upper_bound)).sum()
            if outliers > 0:
                outlier_counts[column] = outliers
        
        if outlier_counts:
            print(f"   Outliers detected per feature: {outlier_counts}")
            
            # Remove rows with outliers in any column
            for column in data_df.columns:
                Q1 = data_df[column].quantile(0.25)
                Q3 = data_df[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - self.outlier_threshold * IQR
                upper_bound = Q3 + self.outlier_threshold * IQR
                
                # Keep only rows within bounds
                data_df = data_df[(data_df[column] >= lower_bound) & (data_df[column] <= upper_bound)]
        
        rows_after = len(data_df)
        rows_removed = rows_before - rows_after
        
        print(f"   Rows removed: {rows_removed} ({rows_removed/rows_before*100:.1f}%)")
        print(f"   Rows remaining: {rows_after}")
        
        return data_df
    
    def apply_pca_reduction(self, data):
        """
        Apply PCA for dimensionality reduction while retaining specified variance
        """
        if data.shape[1] <= 2:
            print("🔧 Skipping PCA: Too few features for dimensionality reduction")
            return data
            
        print(f"🔧 Applying PCA (retaining {self.pca_variance_threshold*100}% variance)...")
        
        features_before = data.shape[1]
        
        # Initialize and fit PCA
        self.pca = PCA(n_components=self.pca_variance_threshold)
        data_pca = self.pca.fit_transform(data)
        
        features_after = data_pca.shape[1]
        variance_explained = self.pca.explained_variance_ratio_.sum()
        
        print(f"   Features reduced: {features_before} → {features_after}")
        print(f"   Variance explained: {variance_explained:.3f} ({variance_explained*100:.1f}%)")
        print(f"   Compression ratio: {features_after/features_before:.2f}x")
        
        return data_pca

    def calculate_portfolio_performance(self, initial_capital=100000, position_size=0.1, max_positions=5):
        """
        Calculate realistic portfolio performance with position sizing and risk management
        
        Args:
            initial_capital: Starting capital in dollars
            position_size: Fraction of capital to risk per trade (default 10%)
            max_positions: Maximum number of concurrent positions
        """
        if not hasattr(self, 'results_df'):
            raise ValueError("Must generate trading signals first")
        
        df = self.results_df.copy()
        portfolio_value = initial_capital
        capital_available = initial_capital
        positions = []  # Track open positions
        portfolio_history = []
        
        for idx, signal in df.iterrows():
            current_date = signal['Date']
            signal_type = signal['Signal_Type']
            confidence = signal['Confidence_Score']
            expected_return = signal['Actual_Gain_Loss_Pct'] / 100
            
            # Only trade high-confidence signals
            if confidence < 2.0:
                continue
                
            # Close expired positions
            positions = [pos for pos in positions if pos['exit_date'] > current_date]
            
            # Calculate current available capital
            capital_in_use = sum(pos['investment'] for pos in positions)
            capital_available = portfolio_value - capital_in_use
            
            # Enter new position if we have room and capital
            if len(positions) < max_positions and capital_available > 0:
                investment_amount = min(capital_available * position_size, capital_available)
                
                # Calculate position details
                entry_price = signal['Entry_Price']
                target_price = signal['Predicted_Price']
                stop_loss_price = entry_price * (0.95 if signal_type == 'BUY' else 1.05)
                
                # Determine exit date and actual return
                duration = signal['Actual_Duration_Days']
                exit_date = current_date + pd.Timedelta(days=duration)
                
                position = {
                    'entry_date': current_date,
                    'exit_date': exit_date,
                    'signal_type': signal_type,
                    'entry_price': entry_price,
                    'target_price': target_price,
                    'stop_loss_price': stop_loss_price,
                    'investment': investment_amount,
                    'shares': investment_amount / entry_price,
                    'expected_return': expected_return,
                    'confidence': confidence
                }
                positions.append(position)
            
            # Calculate portfolio value based on open positions
            portfolio_value = capital_available
            for pos in positions:
                # Use actual returns from our signals
                actual_return = expected_return if pos['entry_date'] <= current_date <= pos['exit_date'] else 0
                position_value = pos['investment'] * (1 + actual_return)
                portfolio_value += position_value
            
            portfolio_history.append({
                'Date': current_date,
                'Portfolio_Value': portfolio_value,
                'Capital_Available': capital_available,
                'Active_Positions': len(positions),
                'Capital_Utilization': (capital_in_use / portfolio_value) * 100 if portfolio_value > 0 else 0
            })
        
        # Calculate final performance metrics
        total_return = (portfolio_value - initial_capital) / initial_capital * 100
        
        portfolio_df = pd.DataFrame(portfolio_history)
        if not portfolio_df.empty:
            portfolio_df['Daily_Return'] = portfolio_df['Portfolio_Value'].pct_change()
            volatility = portfolio_df['Daily_Return'].std() * np.sqrt(252)  # Annualized
            sharpe = (total_return / 100) / volatility if volatility > 0 else 0
            max_drawdown = self._calculate_max_drawdown(portfolio_df['Portfolio_Value'])
        else:
            volatility = 0
            sharpe = 0
            max_drawdown = 0
        
        self.portfolio_performance = {
            'Initial_Capital': initial_capital,
            'Final_Portfolio_Value': portfolio_value,
            'Total_Return_Pct': round(total_return, 2),
            'Annualized_Volatility_Pct': round(volatility * 100, 2),
            'Sharpe_Ratio': round(sharpe, 2),
            'Max_Drawdown_Pct': round(max_drawdown, 2),
            'Portfolio_History': portfolio_df
        }
        
        return self.portfolio_performance
    
    def _calculate_max_drawdown(self, portfolio_values):
        """Calculate maximum drawdown from portfolio values"""
        peak = portfolio_values.expanding().max()
        drawdown = (portfolio_values - peak) / peak
        return drawdown.min() * 100
    
    def optimize_feature_selection(self, X_train, y_train, max_features=20):
        """
        Use advanced feature selection to identify the most predictive indicators
        
        Args:
            X_train: Training features
            y_train: Training targets
            max_features: Maximum number of features to select
        """
        from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
        from sklearn.ensemble import RandomForestRegressor
        
        print(f"Optimizing feature selection from {X_train.shape[1]} indicators...")
        
        # Initialize feature names if not available
        if not hasattr(self, 'feature_columns'):
            self.feature_columns = [f'Feature_{i}' for i in range(X_train.shape[1])]
        
        # Initialize priority indicators if not available  
        if not hasattr(self, 'priority_indicators'):
            self.priority_indicators = ['close', 'RSI', 'MACD', 'ATR', 'OBV']
        
        # Method 1: Statistical F-test
        f_selector = SelectKBest(score_func=f_regression, k=min(max_features, X_train.shape[1]))
        X_f_selected = f_selector.fit_transform(X_train, y_train)
        f_scores = f_selector.scores_
        f_selected_features = f_selector.get_support(indices=True)
        
        # Method 2: Mutual Information
        mi_scores = mutual_info_regression(X_train, y_train)
        mi_selected_features = np.argsort(mi_scores)[-max_features:]
        
        # Method 3: Random Forest Feature Importance
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        rf_importance = rf.feature_importances_
        rf_selected_features = np.argsort(rf_importance)[-max_features:]
        
        # Combine all methods with weighted scoring
        feature_scores = {}
        for i in range(X_train.shape[1]):
            score = 0
            if i in f_selected_features:
                score += f_scores[i] / np.max(f_scores) * 0.3  # 30% weight
            if i in mi_selected_features:
                score += mi_scores[i] / np.max(mi_scores) * 0.3  # 30% weight
            if i in rf_selected_features:
                score += rf_importance[i] / np.max(rf_importance) * 0.4  # 40% weight
            
            feature_scores[i] = score
        
        # Select top features
        selected_indices = sorted(feature_scores.keys(), key=lambda x: feature_scores[x], reverse=True)[:max_features]
        
        # Always include priority features if available
        priority_features = []
        for i, col in enumerate(self.feature_columns):
            if col in self.priority_indicators and i not in selected_indices:
                priority_features.append(i)
        
        # Replace lowest scoring features with priority features
        if priority_features:
            final_features = selected_indices[:-len(priority_features)] + priority_features
        else:
            final_features = selected_indices
        
        self.selected_features = sorted(final_features)
        
        # Store feature importance information
        self.feature_selection_info = {
            'Selected_Features': [self.feature_columns[i] for i in self.selected_features],
            'F_Test_Scores': {self.feature_columns[i]: f_scores[i] for i in range(len(f_scores))},
            'Mutual_Info_Scores': {self.feature_columns[i]: mi_scores[i] for i in range(len(mi_scores))},
            'RF_Importance': {self.feature_columns[i]: rf_importance[i] for i in range(len(rf_importance))},
            'Combined_Scores': {self.feature_columns[i]: feature_scores[i] for i in feature_scores}
        }
        
        print(f"Selected {len(self.selected_features)} optimal features:")
        for feature in self.feature_selection_info['Selected_Features']:
            print(f"  - {feature}")
        
        return X_train[:, self.selected_features]
    
    def evaluate_model_performance(self):
        """
        Comprehensive model evaluation with advanced metrics
        """
        if not hasattr(self, 'y_pred') or not hasattr(self, 'y_actual'):
            raise ValueError("Must make predictions first")
        
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        
        # Basic regression metrics
        mae = mean_absolute_error(self.y_actual, self.y_pred)
        mse = mean_squared_error(self.y_actual, self.y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(self.y_actual, self.y_pred)
        
        # Directional accuracy (how often we predict the right direction)
        actual_direction = np.diff(self.y_actual) > 0
        pred_direction = np.diff(self.y_pred) > 0
        directional_accuracy = np.mean(actual_direction == pred_direction) * 100
        
        # Mean Absolute Percentage Error
        mape = np.mean(np.abs((self.y_actual - self.y_pred) / self.y_actual)) * 100
        
        # Prediction interval analysis
        residuals = self.y_actual - self.y_pred
        prediction_std = np.std(residuals)
        confidence_95 = 1.96 * prediction_std
        
        # Trend prediction accuracy by market conditions
        returns = np.diff(self.y_actual)
        bull_market_mask = returns > 0
        bear_market_mask = returns < 0
        
        bull_directional_acc = np.mean(actual_direction[bull_market_mask] == pred_direction[bull_market_mask]) * 100 if np.any(bull_market_mask) else 0
        bear_directional_acc = np.mean(actual_direction[bear_market_mask] == pred_direction[bear_market_mask]) * 100 if np.any(bear_market_mask) else 0
        
        self.model_evaluation = {
            'MAE': round(mae, 4),
            'MSE': round(mse, 4),
            'RMSE': round(rmse, 4),
            'R2_Score': round(r2, 4),
            'MAPE_Pct': round(mape, 2),
            'Directional_Accuracy_Pct': round(directional_accuracy, 2),
            'Bull_Market_Accuracy_Pct': round(bull_directional_acc, 2),
            'Bear_Market_Accuracy_Pct': round(bear_directional_acc, 2),
            'Prediction_Std': round(prediction_std, 4),
            'Confidence_95_Interval': round(confidence_95, 4)
        }
        
        print("📊 Model Performance Evaluation:")
        for metric, value in self.model_evaluation.items():
            print(f"   {metric}: {value}")
        
        return self.model_evaluation
    
def run_lstm_trading_strategy(data_file="../data/processed/stock_data_with_technical_indicators.csv",
                             company_id=None,
                             target_horizon=5,
                             target_gain=0.10,
                             stop_loss=-0.03,
                             use_advanced_preprocessing=True,
                             outlier_threshold=1.5,
                             pca_variance_threshold=0.95):
    """
    Main function to run the LSTM trading strategy with advanced preprocessing options
    
    Parameters:
    data_file: Path to the technical indicators dataset
    company_id: Specific company to analyze (None = use first company)
    target_horizon: Days to hold position
    target_gain: Target profit percentage
    stop_loss: Stop loss percentage
    use_advanced_preprocessing: Enable advanced preprocessing pipeline
    outlier_threshold: IQR multiplier for outlier detection
    pca_variance_threshold: Minimum variance to retain in PCA
    """
    print("🚀 LSTM TRADING STRATEGY ANALYSIS WITH ADVANCED PREPROCESSING")
    print("=" * 70)
    
    # Load data
    print(f"Loading data from {data_file}...")
    df = pd.read_csv(data_file)
    
    # Filter for specific company if specified
    if company_id is not None and 'companyid' in df.columns:
        df = df[df['companyid'] == company_id]
        print(f"Filtered data for company: {company_id}")
    elif 'companyid' in df.columns:
        # Use the first company if multiple companies exist
        company_id = df['companyid'].iloc[0]
        df = df[df['companyid'] == company_id]
        print(f"Using first company in dataset: {company_id}")
    
    if len(df) < 1000:
        print(f"Warning: Limited data available ({len(df)} rows). Consider using more data for better results.")
    
    # Initialize strategy with advanced preprocessing options
    strategy = LSTMTradingStrategy(
        sequence_length=60,
        target_horizon=target_horizon,
        target_gain=target_gain,
        stop_loss=stop_loss,
        test_split=0.2,
        use_advanced_preprocessing=use_advanced_preprocessing,
        outlier_threshold=outlier_threshold,
        pca_variance_threshold=pca_variance_threshold
    )
    
    # Prepare data (using prioritized technical indicators if available)
    priority_indicators = [
        'close',  # Primary feature (required)
        # Momentum Indicators (Priority)
        'RSI', 'ROC', 'Stoch_K', 'Stoch_D', 'TSI',
        # Volume Indicators (Priority) 
        'OBV', 'MFI', 'PVT',
        # Trend Indicators (Priority)
        'MACD', 'TEMA', 'KAMA', 
        # Volatility Indicators (Priority)
        'ATR', 'BB_Position', 'Ulcer_Index'
    ]
    
    # Check availability and select features
    available_features = ['close']  # Always include close price
    missing_indicators = []
    
    for indicator in priority_indicators[1:]:  # Skip 'close' as it's always available
        if indicator.lower() in [col.lower() for col in df.columns]:
            available_features.append(indicator)
        else:
            missing_indicators.append(indicator)
    
    print(f"Priority indicators analysis:")
    print(f"  ✅ Available indicators ({len(available_features)-1}): {available_features[1:]}")
    if missing_indicators:
        print(f"  ❌ Missing indicators ({len(missing_indicators)}): {missing_indicators}")
        print(f"  💡 Consider regenerating technical indicators to include missing ones")
    
    # If we have fewer than 5 indicators, add some backup indicators
    if len(available_features) < 6:
        backup_indicators = ['EMA_12', 'SMA_20', 'ADX', 'CCI', 'Volume_Ratio']
        for backup in backup_indicators:
            if backup.lower() in [col.lower() for col in df.columns] and backup not in available_features:
                available_features.append(backup)
                if len(available_features) >= 6:
                    break
        print(f"  📊 Added backup indicators: {available_features[6:]}")
    
    print(f"Final feature set ({len(available_features)} features): {available_features}")
    strategy.prepare_data(df, features=available_features)
    
    # Build and train model
    strategy.build_model(lstm_units=[100, 50], dropout_rate=0.2)
    strategy.train_model(epochs=50, batch_size=32, verbose=1)
    
    # Make predictions
    strategy.make_predictions()
    
    # Generate trading signals
    signals_df = strategy.generate_trading_signals(df)
    
    # Calculate summary statistics
    summary = strategy.calculate_summary_statistics()
    
    # Display results
    print("\n📊 TRADING SIGNALS SAMPLE (Enhanced Format):")
    # Generate enhanced evaluation
    enhanced_signals = strategy.generate_enhanced_signal_evaluation()
    
    # Display sample of enhanced signals
    display_cols = ['Date', 'Signal_Type', 'Confidence_Score', 'Actual_Gain_Loss_Pct', 
                   'Trade_Grade', 'Signal_Quality_Score', 'Exit_Reason']
    print(enhanced_signals[display_cols].head(10).to_string(index=False))
    
    print("\n📈 COMPREHENSIVE PERFORMANCE SUMMARY:")
    for key, value in summary.items():
        print(f"  {key.replace('_', ' ')}: {value}")
    
    # Save enhanced results and create comprehensive reports
    enhanced_file, report_path, summary_file = strategy.save_enhanced_results()
    strategy.plot_results()
    
    return strategy, enhanced_signals, summary

if __name__ == "__main__":
    # Run the LSTM trading strategy
    strategy, signals_df, summary = run_lstm_trading_strategy(
        target_horizon=5,      # Hold position for 5 days
        target_gain=0.10,      # Target 10% gain
        stop_loss=-0.03        # Stop loss at -3%
    )
