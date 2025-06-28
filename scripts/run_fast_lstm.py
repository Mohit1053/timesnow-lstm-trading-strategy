# Fast LSTM Trading Strategy Runner
# Optimized for speed while maintaining core functionality

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.lstm_trading_strategy import LSTMTradingStrategy
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

def run_fast_lstm_strategy():
    """
    Run the LSTM trading strategy optimized for speed
    """
    print("⚡ Starting Fast LSTM Trading Strategy")
    print("=" * 50)
    print("🎯 Focus: Fast execution with proven performance")
    
    # Initialize strategy with fast settings (original proven parameters)
    strategy = LSTMTradingStrategy(
        sequence_length=60,        # Keep original proven sequence length
        target_horizon=5,          # Keep original target horizon
        target_gain=0.10,          # Keep original target gain
        stop_loss=-0.03,           # Keep original stop loss
        use_advanced_preprocessing=False,  # DISABLE for speed
        test_split=0.2             # Keep original split
    )
    
    # Load and prepare data
    print("\n📊 Step 1: Loading and preprocessing data...")
    data_path = os.path.abspath("data/processed/stock_data_with_technical_indicators.csv")
    
    if not os.path.exists(data_path):
        print(f"❌ Data file not found: {data_path}")
        print("Please run the technical indicators analysis first.")
        return
    
    # Load the data
    df = pd.read_csv(data_path)
    print(f"Loaded data with shape: {df.shape}")
    
    # Use proven feature set (no complex optimization)
    priority_indicators = [
        'close', 'RSI', 'ROC', 'Stoch_K', 'Stoch_D', 
        'OBV', 'MFI', 'MACD', 'ATR'  # Reduced set for speed
    ]
    
    # Check available features
    available_features = ['close']  # Always include close price
    for indicator in priority_indicators[1:]:
        if indicator.lower() in [col.lower() for col in df.columns]:
            available_features.append(indicator)
        if len(available_features) >= 8:  # Limit to 8 features for speed
            break
    
    print(f"Using {len(available_features)} features for fast processing: {available_features}")
    
    # Prepare data for LSTM (fast mode)
    print("\n⚡ Step 2: Fast data preparation...")
    strategy.prepare_data(df, features=available_features)
    
    # Build model with proven architecture (no optimization)
    print("\n🧠 Step 3: Building LSTM model...")
    strategy.build_model(
        lstm_units=[100, 50],    # Keep original proven architecture  
        dropout_rate=0.2,        # Keep proven dropout rate
        optimize_features=False  # DISABLED for speed
    )
    
    # Train model with fewer epochs for speed
    print("\n🏋️ Step 4: Fast training...")
    strategy.train_model(
        epochs=25,               # Reduced from 50 for speed
        batch_size=32,
        validation_split=0.1,
        verbose=1
    )
    
    # Make predictions
    print("\n🔮 Step 5: Making predictions...")
    strategy.make_predictions()
    
    # Generate trading signals (basic version)
    print("\n🎯 Step 6: Generating trading signals...")
    signals_df = strategy.generate_trading_signals(df)
    
    # Calculate summary statistics
    print("\n📊 Step 7: Calculating summary statistics...")
    summary_stats = strategy.calculate_summary_statistics()
    
    # Generate reports
    print("\n📝 Step 8: Generating reports...")
    output_dir = "output/"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save results
    signals_df.to_csv(f"{output_dir}/fast_trading_signals.csv", index=False)
    
    # Generate plots
    strategy.plot_results(save_path=output_dir)
    
    # Print results
    print("\n" + "=" * 50)
    print("⚡ FAST LSTM TRADING STRATEGY RESULTS")
    print("=" * 50)
    
    print("\n🎯 TRADING PERFORMANCE:")
    for metric, value in summary_stats.items():
        if not metric.startswith('_'):
            print(f"   {metric}: {value}")
    
    # Show sample signals
    print(f"\n📋 SAMPLE TRADING SIGNALS:")
    display_cols = ['Date', 'Signal_Type', 'Confidence_Score', 'Actual_Gain_Loss_Pct', 'Exit_Reason']
    if len(signals_df) > 0:
        print(signals_df[display_cols].head(10).to_string(index=False))
    
    print(f"\n📁 Output files saved to: {os.path.abspath(output_dir)}")
    print("   - fast_trading_signals.csv")
    print("   - lstm_strategy_analysis.png")
    
    print(f"\n⚡ Fast LSTM strategy completed in minimal time!")
    print(f"🎯 Performance: {summary_stats.get('Overall_Accuracy_Pct', 'N/A')}% accuracy")
    print(f"💰 Returns: {summary_stats.get('Total_Return_Pct', 'N/A')}% total return")
    
    return strategy, signals_df, summary_stats

if __name__ == "__main__":
    strategy, signals, summary = run_fast_lstm_strategy()
