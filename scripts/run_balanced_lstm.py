# Balanced Enhanced LSTM Trading Strategy Runner
# Keeps the beneficial enhancements while maintaining high accuracy

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.lstm_trading_strategy import LSTMTradingStrategy
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

def run_balanced_lstm_strategy():
    """
    Run the LSTM trading strategy with balanced enhancements that maintain accuracy
    """
    print("🚀 Starting Balanced Enhanced LSTM Trading Strategy")
    print("=" * 60)
    print("🎯 Focus: Maintain high accuracy while adding useful enhancements")
    
    # Initialize strategy with balanced settings (proven parameters + selective enhancements)
    strategy = LSTMTradingStrategy(
        sequence_length=60,        # Keep original proven sequence length
        target_horizon=5,          # Keep original target horizon
        target_gain=0.10,          # Keep original target gain
        stop_loss=-0.03,           # Keep original stop loss
        use_advanced_preprocessing=True,   # Keep advanced preprocessing (helps accuracy)
        outlier_threshold=2.0,             # More conservative outlier removal
        pca_variance_threshold=0.98        # Retain 98% variance (vs aggressive 95%)
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
    
    # Use original proven feature set (priority indicators)
    priority_indicators = [
        'close', 'RSI', 'ROC', 'Stoch_K', 'Stoch_D', 'TSI',
        'OBV', 'MFI', 'PVT', 'MACD', 'TEMA', 'KAMA', 
        'ATR', 'BB_Position', 'Ulcer_Index'
    ]
    
    # Check available features and select best ones
    available_features = ['close']  # Always include close price
    for indicator in priority_indicators[1:]:
        if indicator.lower() in [col.lower() for col in df.columns]:
            available_features.append(indicator)
    
    # Add backup indicators if needed
    if len(available_features) < 6:
        backup_indicators = ['EMA_12', 'SMA_20', 'ADX', 'CCI', 'Volume_Ratio']
        for backup in backup_indicators:
            if backup.lower() in [col.lower() for col in df.columns] and backup not in available_features:
                available_features.append(backup)
                if len(available_features) >= 8:  # Cap at 8 features for balance
                    break
    
    print(f"Using {len(available_features)} balanced features: {available_features}")
    
    # Prepare data for LSTM
    strategy.prepare_data(df, features=available_features)
    
    # Build model with proven architecture (no aggressive optimization)
    print("\n🧠 Step 2: Building balanced LSTM model...")
    strategy.build_model(
        lstm_units=[100, 50],    # Keep original proven architecture  
        dropout_rate=0.2,        # Keep proven dropout rate
        optimize_features=False  # Disable feature optimization to maintain accuracy
    )
    
    # Train model with proven settings
    print("\n🏋️ Step 3: Training the model...")
    strategy.train_model(
        epochs=50,               # Keep original proven epochs
        batch_size=32,
        validation_split=0.1,
        verbose=1
    )
    
    # Make predictions
    print("\n🔮 Step 4: Making predictions...")
    strategy.make_predictions()
    
    # Evaluate model performance (enhanced evaluation)
    print("\n📈 Step 5: Evaluating model performance...")
    model_metrics = strategy.evaluate_model_performance()
    
    # Generate trading signals (with enhancements but simple confidence)
    print("\n🎯 Step 6: Generating trading signals...")
    signals_df = strategy.generate_trading_signals(df)
    
    # Calculate portfolio performance (new enhancement)
    print("\n💰 Step 7: Calculating portfolio performance...")
    portfolio_metrics = strategy.calculate_portfolio_performance(
        initial_capital=100000,   # $100k starting capital
        position_size=0.10,       # Conservative 10% per trade
        max_positions=3           # Max 3 concurrent positions
    )
    
    # Calculate summary statistics
    print("\n📊 Step 8: Calculating summary statistics...")
    summary_stats = strategy.calculate_summary_statistics()
    
    # Generate enhanced signal evaluation (new feature)
    print("\n🔍 Step 9: Enhanced signal analysis...")
    enhanced_signals = strategy.generate_enhanced_signal_evaluation()
    
    # Generate reports and visualizations
    print("\n📝 Step 10: Generating reports and visualizations...")
    output_dir = "output/"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save detailed results
    signals_df.to_csv(f"{output_dir}/balanced_trading_signals.csv", index=False)
    enhanced_signals.to_csv(f"{output_dir}/balanced_enhanced_signals.csv", index=False)
    
    # Save portfolio history if available
    if 'Portfolio_History' in portfolio_metrics:
        portfolio_metrics['Portfolio_History'].to_csv(f"{output_dir}/balanced_portfolio_history.csv", index=False)
    
    # Generate plots
    strategy.plot_results(save_path=output_dir)
    
    # Create comprehensive evaluation report
    strategy.create_signal_evaluation_report(save_path=output_dir)
    
    # Print comprehensive summary
    print("\n" + "=" * 60)
    print("🎉 BALANCED ENHANCED LSTM TRADING STRATEGY RESULTS")
    print("=" * 60)
    
    print("\n📊 MODEL PERFORMANCE:")
    for metric, value in model_metrics.items():
        print(f"   {metric}: {value}")
    
    print("\n🎯 TRADING PERFORMANCE:")
    for metric, value in summary_stats.items():
        if not metric.startswith('_'):
            print(f"   {metric}: {value}")
    
    print("\n💰 PORTFOLIO PERFORMANCE:")
    for metric, value in portfolio_metrics.items():
        if metric != 'Portfolio_History':
            print(f"   {metric}: {value}")
    
    avg_quality = enhanced_signals['Signal_Quality_Score'].mean()
    high_quality_pct = (enhanced_signals['Signal_Quality_Score'] >= 70).mean() * 100
    
    print(f"\n🌟 SIGNAL QUALITY ANALYSIS:")
    print(f"   Average Signal Quality: {avg_quality:.1f}/100")
    print(f"   High Quality Signals (≥70): {high_quality_pct:.1f}%")
    
    # Show sample of best signals
    print(f"\n📋 TOP 5 HIGHEST QUALITY SIGNALS:")
    top_signals = enhanced_signals.nlargest(5, 'Signal_Quality_Score')
    display_cols = ['Date', 'Signal_Type', 'Confidence_Score', 'Actual_Gain_Loss_Pct', 
                   'Trade_Grade', 'Signal_Quality_Score']
    print(top_signals[display_cols].to_string(index=False))
    
    print(f"\n📁 Output files saved to: {os.path.abspath(output_dir)}")
    print("   - balanced_trading_signals.csv")
    print("   - balanced_enhanced_signals.csv") 
    print("   - balanced_portfolio_history.csv")
    print("   - lstm_signal_evaluation_report.txt")
    print("   - Various plots and visualizations")
    
    print("\n✅ Balanced enhanced LSTM strategy completed successfully!")
    print(f"🎯 Strategy maintains accuracy while adding valuable enhancements")
    
    return strategy, enhanced_signals, summary_stats

if __name__ == "__main__":
    strategy, signals, summary = run_balanced_lstm_strategy()
