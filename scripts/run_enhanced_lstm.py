# Enhanced LSTM Trading Strategy Runner
# This script demonstrates the full capabilities of our advanced trading system

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.lstm_trading_strategy import LSTMTradingStrategy
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

def run_enhanced_lstm_strategy():
    """
    Run the enhanced LSTM trading strategy with all advanced features
    """
    print("🚀 Starting Enhanced LSTM Trading Strategy Pipeline")
    print("=" * 60)
    
    # Initialize strategy with balanced advanced settings
    strategy = LSTMTradingStrategy(
        sequence_length=60,      # Keep original proven length
        target_horizon=5,        # Predict 5 days ahead
        use_advanced_preprocessing=True,   # Keep advanced preprocessing 
        outlier_threshold=2.0,             # More conservative outlier removal
        pca_variance_threshold=0.98        # Retain more variance (98% vs 95%)
    )
    
    # 1. Load and prepare data
    print("\n📊 Step 1: Loading and preprocessing data...")
    data_path = os.path.abspath("data/processed/stock_data_with_technical_indicators.csv")
    
    if not os.path.exists(data_path):
        print(f"❌ Data file not found: {data_path}")
        print("Please run the technical indicators analysis first.")
        return
    
    # Load the data
    df = pd.read_csv(data_path)
    print(f"Loaded data with shape: {df.shape}")
    
    # Prepare features for LSTM
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
    
    print(f"Using {len(available_features)} features: {available_features}")
    
    # Prepare data for LSTM
    strategy.prepare_data(df, features=available_features)
    
    # 2. Build optimized model (conservative approach for better accuracy)
    print("\n🧠 Step 2: Building balanced LSTM model...")
    strategy.build_model(
        lstm_units=[100, 50],    # Keep original proven architecture
        dropout_rate=0.2,        # Standard dropout rate
        optimize_features=False  # Disable feature optimization to maintain accuracy
    )
    
    # 3. Train model (balanced settings)
    print("\n🏋️ Step 3: Training the model...")
    strategy.train_model(
        epochs=50,               # Keep original proven epochs
        batch_size=32,
        validation_split=0.1,    # Standard validation split  
        verbose=1
    )
    
    # 4. Make predictions
    print("\n🔮 Step 4: Making predictions...")
    strategy.make_predictions()
    
    # 5. Evaluate model performance
    print("\n📈 Step 5: Evaluating model performance...")
    model_metrics = strategy.evaluate_model_performance()
    
    # 6. Generate trading signals
    print("\n🎯 Step 6: Generating trading signals...")
    signals_df = strategy.generate_trading_signals(df)
    
    # 7. Calculate portfolio performance
    print("\n💰 Step 7: Calculating portfolio performance...")
    portfolio_metrics = strategy.calculate_portfolio_performance(
        initial_capital=100000,   # $100k starting capital
        position_size=0.15,       # 15% per trade
        max_positions=3           # Max 3 concurrent positions
    )
    
    # 8. Calculate summary statistics
    print("\n📊 Step 8: Calculating summary statistics...")
    summary_stats = strategy.calculate_summary_statistics()
    
    # 9. Generate comprehensive reports
    print("\n📝 Step 9: Generating reports and visualizations...")
    output_dir = "output/"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save detailed results
    signals_df.to_csv(f"{output_dir}/enhanced_trading_signals.csv", index=False)
    
    # Save feature selection results
    if hasattr(strategy, 'feature_selection_info'):
        feature_df = pd.DataFrame(strategy.feature_selection_info['Combined_Scores'].items(), 
                                 columns=['Feature', 'Importance_Score'])
        feature_df = feature_df.sort_values('Importance_Score', ascending=False)
        feature_df.to_csv(f"{output_dir}/feature_importance.csv", index=False)
    
    # Save portfolio history
    if 'Portfolio_History' in portfolio_metrics:
        portfolio_metrics['Portfolio_History'].to_csv(f"{output_dir}/portfolio_history.csv", index=False)
    
    # Generate plots
    strategy.plot_results(save_path=output_dir)
    
    # 10. Print comprehensive summary
    print("\n" + "=" * 60)
    print("🎉 ENHANCED LSTM TRADING STRATEGY RESULTS")
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
    
    if hasattr(strategy, 'feature_selection_info'):
        print(f"\n🔍 SELECTED FEATURES ({len(strategy.feature_selection_info['Selected_Features'])}):")
        for i, feature in enumerate(strategy.feature_selection_info['Selected_Features'][:10], 1):
            print(f"   {i}. {feature}")
        if len(strategy.feature_selection_info['Selected_Features']) > 10:
            print(f"   ... and {len(strategy.feature_selection_info['Selected_Features']) - 10} more")
    
    print(f"\n📁 Output files saved to: {os.path.abspath(output_dir)}")
    print("   - enhanced_trading_signals.csv")
    print("   - feature_importance.csv") 
    print("   - portfolio_history.csv")
    print("   - Various plots and visualizations")
    
    print("\n✅ Enhanced LSTM trading strategy completed successfully!")

if __name__ == "__main__":
    run_enhanced_lstm_strategy()
