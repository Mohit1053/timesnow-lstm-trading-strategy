"""
LSTM Trading Strategy Runner with Enhanced Signal Evaluation

This script runs the LSTM-based trading signal strategy with comprehensive 
signal evaluation including gain/loss analysis, stop losses, accuracy metrics,
and detailed trading performance assessment.

Usage: python run_lstm_strategy.py
"""

import os
import sys

# Add project root and src directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'src'))

# Change to project root directory
os.chdir(project_root)

from src.lstm_trading_strategy import run_lstm_trading_strategy

if __name__ == "__main__":
    print("🤖 LSTM TRADING SIGNAL STRATEGY WITH ENHANCED EVALUATION")
    print("=" * 70)
    print("This enhanced script will:")
    print("✅ Load your technical indicators dataset")
    print("✅ Train an LSTM model to predict price movements")
    print("✅ Generate detailed trading signals with BUY/SELL recommendations")
    print("✅ Evaluate each signal with comprehensive metrics:")
    print("   • Gain/Loss percentage and amounts")
    print("   • Stop loss and target hit analysis") 
    print("   • Signal confidence and quality scores")
    print("   • Risk-adjusted returns and trade grades")
    print("   • Intraday performance tracking")
    print("✅ Create comprehensive evaluation reports and analysis")
    print("✅ Generate enhanced charts and trading performance summaries")
    print()
    
    # Enhanced Strategy Parameters
    COMPANY_ID = None                    # None = use first company, or specify company ID
    TARGET_HORIZON = 5                   # Days to hold position
    TARGET_GAIN = 0.10                   # Target 10% gain
    STOP_LOSS = -0.03                    # Stop loss at -3%
    
    # Advanced Preprocessing Parameters
    USE_ADVANCED_PREPROCESSING = True    # Enable advanced preprocessing pipeline
    OUTLIER_THRESHOLD = 1.5              # IQR multiplier for outlier detection (1.5 = standard)
    PCA_VARIANCE_THRESHOLD = 0.95        # Retain 95% variance in PCA
    
    print(f"Enhanced Strategy Parameters:")
    print(f"  📊 Target Horizon: {TARGET_HORIZON} days")
    print(f"  🎯 Target Gain: {TARGET_GAIN * 100}%")
    print(f"  🛑 Stop Loss: {STOP_LOSS * 100}%")
    print(f"  🏢 Company Filter: {'All companies' if COMPANY_ID is None else COMPANY_ID}")
    print(f"  📈 Portfolio Simulation: Enabled with risk management")
    print(f"  🔍 Signal Quality Scoring: 0-100 scale")
    print()
    print(f"Advanced Preprocessing Pipeline:")
    print(f"  🔧 Linear Interpolation: {'Enabled' if USE_ADVANCED_PREPROCESSING else 'Disabled'}")
    print(f"  📊 IQR Outlier Removal: {'Enabled' if USE_ADVANCED_PREPROCESSING else 'Disabled'} (threshold: {OUTLIER_THRESHOLD})")
    print(f"  🧮 Min-Max Scaling: Enabled")
    print(f"  📉 PCA Dimensionality Reduction: {'Enabled' if USE_ADVANCED_PREPROCESSING else 'Disabled'} (retain {PCA_VARIANCE_THRESHOLD*100}% variance)")
    print()
    
    # Check if technical indicators data exists
    data_file = "data/processed/stock_data_with_technical_indicators.csv"
    if not os.path.exists(data_file):
        print("❌ ERROR: Technical indicators dataset not found!")
        print(f"Expected file: {data_file}")
        print()
        print("Please run the technical indicators analysis first:")
        print("  python run_analysis.py")
        sys.exit(1)
    
    try:
        # Run the LSTM trading strategy with advanced preprocessing
        strategy, signals_df, summary = run_lstm_trading_strategy(
            data_file=data_file,
            company_id=COMPANY_ID,
            target_horizon=TARGET_HORIZON,
            target_gain=TARGET_GAIN,
            stop_loss=STOP_LOSS,
            use_advanced_preprocessing=USE_ADVANCED_PREPROCESSING,
            outlier_threshold=OUTLIER_THRESHOLD,
            pca_variance_threshold=PCA_VARIANCE_THRESHOLD
        )
        
        print("\n🎉 ENHANCED LSTM TRADING STRATEGY COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print("📁 Enhanced output files created:")
        print("  📊 output/lstm_enhanced_trade_signals.csv - Comprehensive trading signals")
        print("  📈 output/lstm_enhanced_strategy_summary.csv - Enhanced performance summary")
        print("  📋 output/lstm_signal_evaluation_report.txt - Detailed evaluation report")
        print("  📊 output/lstm_strategy_analysis.png - Performance charts")
        print()
        
        # Enhanced summary
        print("🔍 ENHANCED RESULTS SUMMARY:")
        print(f"  📊 Total Signals Generated: {summary['Total_Signals']}")
        print(f"  🎯 Overall Accuracy: {summary['Overall_Accuracy_Pct']}%")
        print(f"  💰 Total Return: {summary['Total_Return_Pct']}%")
        print(f"  📈 Win Rate: {summary['Win_Rate_Pct']}%")
        print(f"  ⚖️ Profit Factor: {summary['Profit_Factor']}")
        
        if hasattr(strategy, 'enhanced_results_df'):
            avg_quality = strategy.enhanced_results_df['Signal_Quality_Score'].mean()
            print(f"  🌟 Average Signal Quality: {avg_quality:.1f}/100")
        print()
        
        # Show enhanced sample signals
        print("📋 SAMPLE ENHANCED TRADING SIGNALS:")
        if hasattr(strategy, 'enhanced_results_df'):
            sample_cols = ['Date', 'Signal_Type', 'Confidence_Score', 'Actual_Gain_Loss_Pct', 
                          'Trade_Grade', 'Signal_Quality_Score', 'Exit_Reason']
            print(strategy.enhanced_results_df[sample_cols].head(5).to_string(index=False))
        else:
            sample_cols = ['Date', 'Signal_Type', 'Predicted_Trend', 'Actual_Trend', 
                          'Correct', 'Gain_Pct', 'Duration_Days']
            available_cols = [col for col in sample_cols if col in signals_df.columns]
            print(signals_df[available_cols].head(5).to_string(index=False))
        
        print("\n✅ Enhanced analysis complete! Check the output folder for comprehensive results.")
        print("\n🔍 Recommended next steps:")
        print("1. Review the signal evaluation report for detailed insights")
        print("2. Analyze high-quality signals (Quality Score > 70) for patterns")
        print("3. Consider adjusting parameters based on Grade A/B trade analysis")
        
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        print("\nTroubleshooting:")
        print("1. Ensure you have installed all required packages:")
        print("   pip install -r config/requirements.txt")
        print("2. Make sure you have sufficient data (>1000 rows recommended)")
        print("3. Check that the technical indicators dataset exists")
        sys.exit(1)
