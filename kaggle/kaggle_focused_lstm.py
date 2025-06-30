#!/usr/bin/env python3
"""
Kaggle-Optimized Focused LSTM Trading Strategy

📊 COMPLETE SELF-CONTAINED SCRIPT FOR KAGGLE EXECUTION
This script runs an LSTM trading strategy using only 13 specific technical indicators.
Designed for direct execution in Kaggle notebooks or script environments.

🚀 USAGE IN KAGGLE:
1. Upload your CSV data file to Kaggle dataset
2. Update DATA_PATH below to match your data location
3. Run this script directly in a Kaggle notebook or code cell

📈 Indicators Used (13 Total):
- Momentum: RSI, ROC, Stochastic Oscillator, TSI
- Volume: OBV, MFI, PVT  
- Trend: TEMA, MACD, KAMA
- Volatility: ATR, Bollinger Bands, Ulcer Index

📁 Outputs Generated:
- focused_lstm_results.csv: Trading signals and predictions
- focused_trading_log.csv: Detailed trading log
- focused_portfolio_summary.txt: Performance summary
- focused_model_metrics.txt: Model evaluation metrics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Install required packages if running in Kaggle
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    import talib
except ImportError:
    import subprocess
    import sys
    
    packages = [
        'tensorflow==2.13.0',
        'scikit-learn',
        'TA-Lib',
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
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    import talib

# ============================================================================
# CONFIGURATION - UPDATE THESE PATHS FOR YOUR KAGGLE ENVIRONMENT
# ============================================================================

# Path to your data file in Kaggle
# For Kaggle datasets: '/kaggle/input/your-dataset-name/filename.csv'
# For uploaded files: '/kaggle/working/filename.csv'
DATA_PATH = '/kaggle/input/timesnow-price-data/priceData5Year.csv'  # UPDATE THIS PATH

# Output directory (Kaggle working directory)
OUTPUT_DIR = '/kaggle/working'

# Model configuration
LSTM_CONFIG = {
    'sequence_length': 60,     # Days of historical data for prediction
    'lstm_units': 50,          # LSTM layer units
    'dropout_rate': 0.2,       # Dropout for regularization
    'epochs': 50,              # Training epochs (reduced for Kaggle)
    'batch_size': 32,          # Batch size
    'validation_split': 0.2,   # Validation data percentage
    'learning_rate': 0.001     # Adam optimizer learning rate
}

# Trading configuration
TRADING_CONFIG = {
    'initial_capital': 100000,  # Starting capital
    'transaction_cost': 0.001,  # 0.1% transaction cost
    'position_sizing': 'fixed', # 'fixed' or 'dynamic'
    'max_position_size': 0.95,  # Maximum portfolio allocation per trade
    'stop_loss': 0.05,          # 5% stop loss
    'take_profit': 0.15         # 15% take profit
}

# ============================================================================
# TECHNICAL INDICATORS - FOCUSED 13 INDICATORS
# ============================================================================

def calculate_focused_indicators(df):
    """Calculate the 13 specific technical indicators for the LSTM model."""
    data = df.copy()
    
    print("📊 Calculating 13 focused technical indicators...")
    
    # Ensure we have OHLCV columns
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in required_cols:
        if col not in data.columns:
            raise ValueError(f"Required column '{col}' not found in data")
    
    # Convert to numpy arrays for TA-Lib
    open_prices = data['Open'].astype(float).values
    high_prices = data['High'].astype(float).values
    low_prices = data['Low'].astype(float).values
    close_prices = data['Close'].astype(float).values
    volume = data['Volume'].astype(float).values
    
    # 1. MOMENTUM INDICATORS (4)
    print("   📈 Momentum indicators...")
    
    # RSI - Relative Strength Index
    data['RSI'] = talib.RSI(close_prices, timeperiod=14)
    
    # ROC - Rate of Change
    data['ROC'] = talib.ROC(close_prices, timeperiod=10)
    
    # Stochastic Oscillator
    data['STOCH_K'], data['STOCH_D'] = talib.STOCH(high_prices, low_prices, close_prices)
    
    # TSI - True Strength Index (using TRIX as approximation)
    data['TSI'] = talib.TRIX(close_prices, timeperiod=14)
    
    # 2. VOLUME INDICATORS (3)
    print("   📊 Volume indicators...")
    
    # OBV - On Balance Volume
    data['OBV'] = talib.OBV(close_prices, volume)
    
    # MFI - Money Flow Index
    data['MFI'] = talib.MFI(high_prices, low_prices, close_prices, volume)
    
    # PVT - Price Volume Trend (manual calculation)
    data['PVT'] = ((close_prices - np.roll(close_prices, 1)) / np.roll(close_prices, 1) * volume).cumsum()
    
    # 3. TREND INDICATORS (3)
    print("   📉 Trend indicators...")
    
    # TEMA - Triple Exponential Moving Average
    data['TEMA'] = talib.TEMA(close_prices, timeperiod=14)
    
    # MACD
    data['MACD'], data['MACD_Signal'], data['MACD_Hist'] = talib.MACD(close_prices)
    
    # KAMA - Kaufman's Adaptive Moving Average
    data['KAMA'] = talib.KAMA(close_prices, timeperiod=30)
    
    # 4. VOLATILITY INDICATORS (3)
    print("   🌪️ Volatility indicators...")
    
    # ATR - Average True Range
    data['ATR'] = talib.ATR(high_prices, low_prices, close_prices)
    
    # Bollinger Bands
    data['BB_Upper'], data['BB_Middle'], data['BB_Lower'] = talib.BBANDS(close_prices)
    data['BB_Width'] = data['BB_Upper'] - data['BB_Lower']
    data['BB_Position'] = (close_prices - data['BB_Lower']) / (data['BB_Upper'] - data['BB_Lower'])
    
    # Ulcer Index (manual calculation)
    def calculate_ulcer_index(prices, period=14):
        max_prices = pd.Series(prices).rolling(window=period).max()
        drawdown = ((prices - max_prices) / max_prices) * 100
        ulcer = np.sqrt((drawdown ** 2).rolling(window=period).mean())
        return ulcer.values
    
    data['ULCER'] = calculate_ulcer_index(close_prices)
    
    print(f"✅ Successfully calculated {13} technical indicators")
    
    return data

# ============================================================================
# LSTM TRADING STRATEGY CLASS
# ============================================================================

class KaggleLSTMStrategy:
    """LSTM Trading Strategy optimized for Kaggle execution."""
    
    def __init__(self, config=LSTM_CONFIG):
        self.config = config
        self.model = None
        self.scaler = MinMaxScaler()
        self.feature_columns = [
            'RSI', 'ROC', 'STOCH_K', 'STOCH_D', 'TSI',  # Momentum
            'OBV', 'MFI', 'PVT',                         # Volume
            'TEMA', 'MACD', 'MACD_Signal', 'MACD_Hist', 'KAMA',  # Trend
            'ATR', 'BB_Width', 'BB_Position', 'ULCER'    # Volatility
        ]
        
    def prepare_data(self, df):
        """Prepare data for LSTM training."""
        print("🔄 Preparing data for LSTM...")
        
        # Select feature columns and target
        data = df[self.feature_columns + ['Close']].copy()
        
        # Remove NaN values
        data = data.dropna()
        
        # Scale features
        scaled_data = self.scaler.fit_transform(data)
        
        # Create sequences
        X, y = [], []
        sequence_length = self.config['sequence_length']
        
        for i in range(sequence_length, len(scaled_data)):
            X.append(scaled_data[i-sequence_length:i, :-1])  # All features except Close
            y.append(scaled_data[i, -1])  # Close price
        
        return np.array(X), np.array(y), data.index[sequence_length:]
    
    def build_model(self, input_shape):
        """Build LSTM model."""
        print("🏗️ Building LSTM model...")
        
        model = Sequential([
            LSTM(self.config['lstm_units'], 
                 return_sequences=True, 
                 input_shape=input_shape),
            Dropout(self.config['dropout_rate']),
            
            LSTM(self.config['lstm_units'] // 2, 
                 return_sequences=False),
            Dropout(self.config['dropout_rate']),
            
            Dense(25),
            Dense(1)
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=self.config['learning_rate']),
            loss='mean_squared_error'
        )
        
        return model
    
    def train(self, X, y):
        """Train the LSTM model."""
        print("🎯 Training LSTM model...")
        
        # Build model
        self.model = self.build_model((X.shape[1], X.shape[2]))
        
        # Train model
        history = self.model.fit(
            X, y,
            epochs=self.config['epochs'],
            batch_size=self.config['batch_size'],
            validation_split=self.config['validation_split'],
            verbose=1
        )
        
        return history
    
    def predict(self, X):
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict(X)
    
    def evaluate_model(self, y_true, y_pred):
        """Evaluate model performance."""
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        
        return {
            'MSE': mse,
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2
        }

# ============================================================================
# TRADING SIMULATION
# ============================================================================

def simulate_trading(df, predictions, config=TRADING_CONFIG):
    """Simulate trading strategy based on LSTM predictions."""
    print("💼 Simulating trading strategy...")
    
    results = df.copy()
    results['Prediction'] = predictions
    results['Signal'] = 0
    results['Position'] = 0
    results['Portfolio_Value'] = config['initial_capital']
    results['Returns'] = 0
    
    # Generate trading signals
    results.loc[results['Prediction'] > results['Close'].shift(1), 'Signal'] = 1  # Buy
    results.loc[results['Prediction'] < results['Close'].shift(1), 'Signal'] = -1  # Sell
    
    # Simulate trading
    capital = config['initial_capital']
    position = 0
    shares = 0
    
    trading_log = []
    
    for i in range(1, len(results)):
        current_price = results.iloc[i]['Close']
        signal = results.iloc[i]['Signal']
        
        # Calculate transaction cost
        transaction_cost = current_price * abs(signal) * config['transaction_cost']
        
        if signal == 1 and position <= 0:  # Buy signal
            # Calculate position size
            max_investment = capital * config['max_position_size']
            shares_to_buy = max_investment // current_price
            cost = shares_to_buy * current_price + transaction_cost
            
            if cost <= capital:
                capital -= cost
                shares += shares_to_buy
                position = 1
                
                trading_log.append({
                    'Date': results.index[i],
                    'Action': 'BUY',
                    'Price': current_price,
                    'Shares': shares_to_buy,
                    'Cost': cost,
                    'Capital': capital
                })
        
        elif signal == -1 and position >= 0 and shares > 0:  # Sell signal
            revenue = shares * current_price - transaction_cost
            capital += revenue
            
            trading_log.append({
                'Date': results.index[i],
                'Action': 'SELL',
                'Price': current_price,
                'Shares': shares,
                'Revenue': revenue,
                'Capital': capital
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
            returns = (portfolio_value - prev_value) / prev_value
            results.iloc[i, results.columns.get_loc('Returns')] = returns
    
    return results, pd.DataFrame(trading_log)

# ============================================================================
# PERFORMANCE ANALYSIS
# ============================================================================

def analyze_performance(results, trading_log, initial_capital):
    """Analyze trading performance."""
    print("📊 Analyzing performance...")
    
    final_value = results['Portfolio_Value'].iloc[-1]
    total_return = (final_value - initial_capital) / initial_capital
    
    # Calculate various metrics
    returns = results['Returns'].dropna()
    
    # Sharpe ratio (assuming 0% risk-free rate)
    sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
    
    # Maximum drawdown
    cumulative_returns = (1 + returns).cumprod()
    running_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # Win rate
    profitable_trades = trading_log[trading_log['Action'] == 'SELL']
    if len(profitable_trades) > 0:
        # This is simplified - would need buy/sell pairs for accurate calculation
        win_rate = 0.5  # Placeholder
    else:
        win_rate = 0
    
    metrics = {
        'Total Return': f"{total_return:.2%}",
        'Final Portfolio Value': f"${final_value:,.2f}",
        'Initial Capital': f"${initial_capital:,.2f}",
        'Profit/Loss': f"${final_value - initial_capital:,.2f}",
        'Sharpe Ratio': f"{sharpe_ratio:.3f}",
        'Maximum Drawdown': f"{max_drawdown:.2%}",
        'Total Trades': len(trading_log),
        'Average Daily Return': f"{returns.mean():.4f}",
        'Volatility': f"{returns.std():.4f}"
    }
    
    return metrics

# ============================================================================
# VISUALIZATION
# ============================================================================

def create_visualizations(results, trading_log):
    """Create performance visualizations."""
    print("📈 Creating visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Price vs Predictions
    axes[0, 0].plot(results.index, results['Close'], label='Actual Price', alpha=0.7)
    axes[0, 0].plot(results.index, results['Prediction'], label='LSTM Prediction', alpha=0.7)
    axes[0, 0].set_title('Price vs LSTM Predictions')
    axes[0, 0].legend()
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 2. Portfolio Value
    axes[0, 1].plot(results.index, results['Portfolio_Value'])
    axes[0, 1].set_title('Portfolio Value Over Time')
    axes[0, 1].set_ylabel('Portfolio Value ($)')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # 3. Trading Signals
    buy_signals = results[results['Signal'] == 1]
    sell_signals = results[results['Signal'] == -1]
    
    axes[1, 0].plot(results.index, results['Close'], label='Price', alpha=0.7)
    axes[1, 0].scatter(buy_signals.index, buy_signals['Close'], 
                       color='green', marker='^', label='Buy', s=50)
    axes[1, 0].scatter(sell_signals.index, sell_signals['Close'], 
                       color='red', marker='v', label='Sell', s=50)
    axes[1, 0].set_title('Trading Signals')
    axes[1, 0].legend()
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 4. Returns Distribution
    returns = results['Returns'].dropna()
    axes[1, 1].hist(returns, bins=50, alpha=0.7)
    axes[1, 1].set_title('Returns Distribution')
    axes[1, 1].set_xlabel('Daily Returns')
    axes[1, 1].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/focused_lstm_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

# ============================================================================
# MAIN EXECUTION FUNCTION
# ============================================================================

def main():
    """Main execution function for Kaggle LSTM trading strategy."""
    print("🚀 KAGGLE FOCUSED LSTM TRADING STRATEGY")
    print("=" * 60)
    print(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    try:
        # 1. Load data
        print("📂 Loading data...")
        df = pd.read_csv(DATA_PATH)
        
        # Ensure Date column is datetime and set as index
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
        elif df.index.name != 'Date':
            df.index = pd.to_datetime(df.index)
        
        print(f"   📊 Data shape: {df.shape}")
        print(f"   📅 Date range: {df.index.min()} to {df.index.max()}")
        print()
        
        # 2. Calculate technical indicators
        df_with_indicators = calculate_focused_indicators(df)
        
        # 3. Initialize and prepare LSTM strategy
        strategy = KaggleLSTMStrategy()
        X, y, dates = strategy.prepare_data(df_with_indicators)
        
        print(f"   🔢 Training samples: {X.shape[0]}")
        print(f"   📏 Sequence length: {X.shape[1]}")
        print(f"   📊 Features: {X.shape[2]}")
        print()
        
        # 4. Train LSTM model
        history = strategy.train(X, y)
        
        # 5. Make predictions
        print("🔮 Making predictions...")
        predictions_scaled = strategy.predict(X)
        
        # Inverse transform predictions
        dummy_data = np.zeros((len(predictions_scaled), len(strategy.feature_columns) + 1))
        dummy_data[:, -1] = predictions_scaled.flatten()
        predictions = strategy.scaler.inverse_transform(dummy_data)[:, -1]
        
        # 6. Evaluate model
        actual_prices = df_with_indicators.loc[dates, 'Close'].values
        metrics = strategy.evaluate_model(actual_prices, predictions)
        
        print("📊 Model Performance:")
        for metric, value in metrics.items():
            print(f"   {metric}: {value:.6f}")
        print()
        
        # 7. Simulate trading
        results_df = df_with_indicators.loc[dates].copy()
        results, trading_log = simulate_trading(results_df, predictions)
        
        # 8. Analyze performance
        performance_metrics = analyze_performance(results, trading_log, TRADING_CONFIG['initial_capital'])
        
        print("💰 Trading Performance:")
        for metric, value in performance_metrics.items():
            print(f"   {metric}: {value}")
        print()
        
        # 9. Save results
        print("💾 Saving results...")
        
        # Save main results
        output_file = f"{OUTPUT_DIR}/focused_lstm_results.csv"
        results.to_csv(output_file)
        print(f"   ✅ Results saved to: {output_file}")
        
        # Save trading log
        log_file = f"{OUTPUT_DIR}/focused_trading_log.csv"
        trading_log.to_csv(log_file, index=False)
        print(f"   ✅ Trading log saved to: {log_file}")
        
        # Save performance summary
        summary_file = f"{OUTPUT_DIR}/focused_portfolio_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("FOCUSED LSTM TRADING STRATEGY - PERFORMANCE SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Data Period: {df.index.min()} to {df.index.max()}\n\n")
            
            f.write("TRADING PERFORMANCE:\n")
            f.write("-" * 30 + "\n")
            for metric, value in performance_metrics.items():
                f.write(f"{metric}: {value}\n")
            
            f.write("\nMODEL METRICS:\n")
            f.write("-" * 30 + "\n")
            for metric, value in metrics.items():
                f.write(f"{metric}: {value:.6f}\n")
            
            f.write("\nCONFIGURATION:\n")
            f.write("-" * 30 + "\n")
            f.write(f"LSTM Units: {LSTM_CONFIG['lstm_units']}\n")
            f.write(f"Sequence Length: {LSTM_CONFIG['sequence_length']}\n")
            f.write(f"Training Epochs: {LSTM_CONFIG['epochs']}\n")
            f.write(f"Initial Capital: ${TRADING_CONFIG['initial_capital']:,}\n")
            f.write(f"Transaction Cost: {TRADING_CONFIG['transaction_cost']:.1%}\n")
        
        print(f"   ✅ Summary saved to: {summary_file}")
        
        # Save model metrics
        metrics_file = f"{OUTPUT_DIR}/focused_model_metrics.txt"
        with open(metrics_file, 'w') as f:
            f.write("LSTM MODEL EVALUATION METRICS\n")
            f.write("=" * 40 + "\n\n")
            for metric, value in metrics.items():
                f.write(f"{metric}: {value:.6f}\n")
        
        print(f"   ✅ Model metrics saved to: {metrics_file}")
        
        # 10. Create visualizations
        create_visualizations(results, trading_log)
        
        print("\n🎉 ANALYSIS COMPLETE!")
        print("=" * 60)
        print("📊 Generated Files:")
        print(f"   • {output_file}")
        print(f"   • {log_file}")
        print(f"   • {summary_file}")
        print(f"   • {metrics_file}")
        print(f"   • {OUTPUT_DIR}/focused_lstm_analysis.png")
        print()
        print("💡 Next Steps:")
        print("   • Review the generated CSV files for detailed results")
        print("   • Check the PNG chart for visual analysis")
        print("   • Modify parameters in the config section to optimize performance")
        print("   • Consider using more historical data for better predictions")
        
    except Exception as e:
        print(f"❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

# ============================================================================
# KAGGLE EXECUTION
# ============================================================================

if __name__ == "__main__":
    main()
