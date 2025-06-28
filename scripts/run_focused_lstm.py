#!/usr/bin/env python3
"""
Focused LSTM Trading Strategy - Using Only 13 Specific Indicators

This script runs an LSTM trading strategy using only the specified indicators:

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

This script prioritizes speed and simplicity while maintaining effectiveness.
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from lstm_trading_strategy import LSTMTradingStrategy

def load_price_data(file_path):
    """Load and validate processed price data with technical indicators"""
    print(f"📊 Loading processed data from: {file_path}")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    df = pd.read_csv(file_path)
    
    # Check if data has an index column or date column
    if 'Unnamed: 0' in df.columns:
        df.set_index('Unnamed: 0', inplace=True)
        df.index = pd.to_datetime(df.index)
    elif 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
    elif 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
    
    df.sort_index(inplace=True)
    
    print(f"✅ Loaded {len(df)} records with {len(df.columns)} columns")
    print(f"📅 Data range: {df.index[0].date()} to {df.index[-1].date()}")
    print(f"📊 Available columns: {list(df.columns)[:10]}..." if len(df.columns) > 10 else f"📊 Available columns: {list(df.columns)}")
    
    return df

def extract_focused_indicators(df):
    """Extract only the 13 specified indicators from processed data"""
    print("🔧 Extracting focused technical indicators...")
    
    # Define the 13 focused indicators we want
    focused_indicators = {
        # 📈 Momentum Indicators
        'RSI': 'RSI',
        'ROC': 'ROC', 
        'Stoch_K': 'Stoch_K',
        'Stoch_D': 'Stoch_D',
        'TSI': 'TSI',
        
        # 📊 Volume Indicators  
        'OBV': 'OBV',
        'MFI': 'MFI',
        'PVT': 'PVT',
        
        # � Trend Indicators
        'TEMA': 'TEMA',
        'MACD': 'MACD',
        'MACD_Signal': 'MACD_Signal',
        'MACD_Histogram': 'MACD_Histogram',
        'KAMA': 'KAMA',
        
        # 🌪️ Volatility Indicators
        'ATR': 'ATR',
        'BB_Upper': 'BB_Upper',
        'BB_Middle': 'BB_Middle', 
        'BB_Lower': 'BB_Lower',
        'Ulcer_Index': 'Ulcer_Index'
    }
    
    # Start with basic OHLCV data
    basic_columns = ['open', 'high', 'low', 'close', 'volume']
    result_df = pd.DataFrame(index=df.index)
    
    # Copy basic OHLCV data (handle different column name cases)
    for col in basic_columns:
        found = False
        for df_col in df.columns:
            if df_col.lower() == col:
                result_df[col.title()] = df[df_col]
                found = True
                break
        if not found:
            print(f"⚠️ Warning: {col} column not found in data")
    
    # Extract focused indicators
    extracted_count = 0
    missing_indicators = []
    
    for indicator_name, column_name in focused_indicators.items():
        if column_name in df.columns:
            result_df[indicator_name] = df[column_name]
            extracted_count += 1
        else:
            missing_indicators.append(column_name)
            print(f"⚠️ Warning: {column_name} not found in processed data")
    
    # Add computed Bollinger Band indicators if we have the components
    if 'BB_Upper' in result_df.columns and 'BB_Lower' in result_df.columns and 'Close' in result_df.columns:
        result_df['BB_Width'] = result_df['BB_Upper'] - result_df['BB_Lower']
        result_df['BB_Position'] = ((result_df['Close'] - result_df['BB_Lower']) / 
                                   (result_df['BB_Upper'] - result_df['BB_Lower']))
        extracted_count += 2
    
    print(f"✅ Extracted {extracted_count} focused indicators")
    if missing_indicators:
        print(f"⚠️ Missing indicators: {missing_indicators}")
    
    # Count final indicator columns (excluding OHLCV)
    indicator_columns = [col for col in result_df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume']]
    print(f"📊 Final dataset has {len(indicator_columns)} technical indicators")
    
    return result_df

def run_focused_lstm_strategy():
    """Run the focused LSTM trading strategy"""
    print("=" * 80)
    print("🎯 FOCUSED LSTM TRADING STRATEGY")
    print("=" * 80)
    print("Using only 13 specific indicators for optimal speed and focus")
    print()
    
    # Setup paths
    base_dir = os.path.dirname(os.path.dirname(__file__))
    data_file = os.path.join(base_dir, 'data', 'processed', 'stock_data_with_technical_indicators.csv')
    output_dir = os.path.join(base_dir, 'output')
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Load price data
        df = load_price_data(data_file)
        
        # Extract focused indicators
        df_with_indicators = extract_focused_indicators(df)
        
        # Initialize LSTM strategy with focused settings
        print("\n🤖 Initializing LSTM Trading Strategy...")
        strategy = LSTMTradingStrategy(
            sequence_length=60,                  # Proven length
            target_horizon=5,                    # 5-day prediction
            use_advanced_preprocessing=True,     # Keep preprocessing for stability
            outlier_threshold=2.0,              # Conservative outlier removal
            pca_variance_threshold=0.98         # Conservative PCA
        )
        
        # Prepare data with minimal preprocessing
        print("📋 Preparing data for LSTM...")
        X_train, X_test, y_train, y_test = strategy.prepare_data(df_with_indicators)
        
        print(f"✅ Prepared data shapes:")
        print(f"   Training: X={X_train.shape}, y={y_train.shape}")
        print(f"   Testing: X={X_test.shape}, y={y_test.shape}")
        
        # Build and train model
        print("\n🏗️ Building LSTM model...")
        strategy.build_model(
            lstm_units=[100, 50],               # Proven architecture
            dropout_rate=0.2,                   # Proven dropout
            optimize_features=False             # No feature optimization for speed
        )
        
        print("🎓 Training LSTM model...")
        history = strategy.train_model(epochs=50, batch_size=32, validation_split=0.2)
        
        # Generate predictions and signals
        print("\n📈 Generating trading signals...")
        signals_df = strategy.generate_trading_signals(df_with_indicators)
        
        # Evaluate strategy
        print("\n📊 Evaluating strategy performance...")
        evaluation_results = strategy.evaluate_strategy(df_with_indicators, signals_df)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save signals
        signals_file = os.path.join(output_dir, f'focused_lstm_signals_{timestamp}.csv')
        signals_df.to_csv(signals_file)
        print(f"💾 Signals saved to: {signals_file}")
        
        # Save evaluation report
        report_file = os.path.join(output_dir, f'focused_lstm_report_{timestamp}.txt')
        with open(report_file, 'w') as f:
            f.write("FOCUSED LSTM TRADING STRATEGY REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("INDICATORS USED:\n")
            f.write("-" * 20 + "\n")
            f.write("📈 Momentum: RSI, ROC, Stochastic Oscillator, TSI\n")
            f.write("📊 Volume: OBV, MFI, PVT\n")
            f.write("📉 Trend: TEMA, MACD, KAMA\n")
            f.write("🌪️ Volatility: ATR, Bollinger Bands, Ulcer Index\n\n")
            
            f.write("STRATEGY CONFIGURATION:\n")
            f.write("-" * 25 + "\n")
            f.write(f"Sequence Length: {strategy.sequence_length}\n")
            f.write(f"Target Horizon: {strategy.target_horizon} days\n")
            f.write(f"LSTM Units: [100, 50]\n")
            f.write(f"Dropout Rate: 0.2\n")
            f.write(f"Advanced Preprocessing: Yes\n")
            f.write(f"Feature Optimization: No (for speed)\n\n")
            
            f.write("PERFORMANCE METRICS:\n")
            f.write("-" * 20 + "\n")
            for key, value in evaluation_results.items():
                if isinstance(value, dict):
                    f.write(f"{key}:\n")
                    for k, v in value.items():
                        f.write(f"  {k}: {v}\n")
                else:
                    f.write(f"{key}: {value}\n")
        
        print(f"📄 Report saved to: {report_file}")
        
        # Print summary
        print("\n" + "=" * 50)
        print("🎯 FOCUSED LSTM STRATEGY SUMMARY")
        print("=" * 50)
        
        if 'accuracy_metrics' in evaluation_results:
            accuracy = evaluation_results['accuracy_metrics'].get('overall_accuracy', 'N/A')
            print(f"📊 Overall Accuracy: {accuracy}")
        
        if 'portfolio_performance' in evaluation_results:
            portfolio = evaluation_results['portfolio_performance']
            total_return = portfolio.get('total_return_pct', 'N/A')
            sharpe_ratio = portfolio.get('sharpe_ratio', 'N/A')
            max_drawdown = portfolio.get('max_drawdown_pct', 'N/A')
            
            print(f"💰 Total Return: {total_return}")
            print(f"📈 Sharpe Ratio: {sharpe_ratio}")
            print(f"📉 Max Drawdown: {max_drawdown}")
        
        if 'signal_quality' in evaluation_results:
            signal_quality = evaluation_results['signal_quality']
            avg_confidence = signal_quality.get('average_confidence', 'N/A')
            high_conf_signals = signal_quality.get('high_confidence_signals', 'N/A')
            
            print(f"🎯 Average Confidence: {avg_confidence}")
            print(f"⭐ High Confidence Signals: {high_conf_signals}")
        
        print("\n✅ Focused LSTM strategy completed successfully!")
        print(f"📁 All results saved to: {output_dir}")
        
    except Exception as e:
        print(f"\n❌ Error running focused LSTM strategy: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = run_focused_lstm_strategy()
    sys.exit(0 if success else 1)
