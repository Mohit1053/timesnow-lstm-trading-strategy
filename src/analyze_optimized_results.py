import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("=== OPTIMIZED LSTM STRATEGY ANALYSIS ===")
print("=" * 60)

# Read the optimized strategy results
df = pd.read_csv('output/optimized_strategy_results_20250706_144340.csv')

print("1. BASIC STATISTICS:")
print(f"Total signals: {len(df)}")
print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
print(f"Companies: {df['Company'].unique()}")
print()

# Overall performance
correct_predictions = (df['Correct_Prediction'] == 1).sum()
total_predictions = len(df)
accuracy = correct_predictions / total_predictions * 100

print("2. OVERALL PERFORMANCE:")
print(f"Correct predictions: {correct_predictions}")
print(f"Total predictions: {total_predictions}")
print(f"Overall accuracy: {accuracy:.2f}%")
print()

# Warning analysis
warning_counts = df['Warning'].value_counts()
opposite_count = warning_counts.get('OPPOSITE_PREDICTION', 0)
correct_count = warning_counts.get('CORRECT_PREDICTION', 0)

print("3. PREDICTION QUALITY:")
print(f"Correct predictions: {correct_count} ({correct_count/len(df)*100:.1f}%)")
print(f"Opposite predictions: {opposite_count} ({opposite_count/len(df)*100:.1f}%)")
print()

# Signal analysis
signal_counts = df['Predicted_Signal'].value_counts()
print("4. SIGNAL DISTRIBUTION:")
print(f"Bullish signals (1): {signal_counts.get(1, 0)} ({signal_counts.get(1, 0)/len(df)*100:.1f}%)")
print(f"Bearish signals (0): {signal_counts.get(0, 0)} ({signal_counts.get(0, 0)/len(df)*100:.1f}%)")
print()

# Signal performance
bullish_data = df[df['Predicted_Signal'] == 1]
bearish_data = df[df['Predicted_Signal'] == 0]

print("5. SIGNAL PERFORMANCE:")
if len(bullish_data) > 0:
    bullish_accuracy = bullish_data['Correct_Prediction'].mean()
    print(f"Bullish signal accuracy: {bullish_accuracy:.2%}")
    
if len(bearish_data) > 0:
    bearish_accuracy = bearish_data['Correct_Prediction'].mean()
    print(f"Bearish signal accuracy: {bearish_accuracy:.2%}")
print()

# Confidence analysis
print("6. CONFIDENCE ANALYSIS:")
print(f"Average confidence: {df['Confidence'].mean():.4f}")
print(f"Confidence std: {df['Confidence'].std():.4f}")
print(f"Min confidence: {df['Confidence'].min():.4f}")
print(f"Max confidence: {df['Confidence'].max():.4f}")

# High confidence signals
high_conf_threshold = 0.7
high_conf_signals = df[df['Confidence'] >= high_conf_threshold]
if len(high_conf_signals) > 0:
    high_conf_accuracy = high_conf_signals['Correct_Prediction'].mean()
    print(f"High confidence signals (>= {high_conf_threshold}): {len(high_conf_signals)} ({len(high_conf_signals)/len(df)*100:.1f}%)")
    print(f"High confidence accuracy: {high_conf_accuracy:.2%}")
print()

# Signal strength analysis
print("7. SIGNAL STRENGTH ANALYSIS:")
print(f"Average signal strength: {df['Signal_Strength'].mean():.4f}")
print(f"Signal strength std: {df['Signal_Strength'].std():.4f}")
print(f"Min signal strength: {df['Signal_Strength'].min():.4f}")
print(f"Max signal strength: {df['Signal_Strength'].max():.4f}")
print()

# Returns analysis
print("8. RETURNS ANALYSIS:")
print(f"Average actual return: {df['Actual_Return'].mean():.4f}")
print(f"Return volatility: {df['Actual_Return'].std():.4f}")
print(f"Positive returns: {(df['Actual_Return'] > 0).sum()} ({(df['Actual_Return'] > 0).mean():.1%})")
print(f"Negative returns: {(df['Actual_Return'] < 0).sum()} ({(df['Actual_Return'] < 0).mean():.1%})")
print()

# Performance by signal probability ranges
print("9. PERFORMANCE BY SIGNAL PROBABILITY:")
prob_bins = [0, 0.5, 0.6, 0.7, 0.8, 1.0]
prob_labels = ['Low (0-0.5)', 'Medium (0.5-0.6)', 'High (0.6-0.7)', 'Very High (0.7-0.8)', 'Extreme (0.8-1.0)']

df['Prob_Bin'] = pd.cut(df['Signal_Probability'], bins=prob_bins, labels=prob_labels, include_lowest=True)
prob_performance = df.groupby('Prob_Bin').agg({
    'Correct_Prediction': ['count', 'sum', 'mean'],
    'Signal_Strength': 'mean',
    'Confidence': 'mean'
}).round(4)

prob_performance.columns = ['Count', 'Correct', 'Accuracy', 'Avg_Strength', 'Avg_Confidence']
print(prob_performance)
print()

# Time-based analysis
print("10. TIME-BASED ANALYSIS:")
df['Date'] = pd.to_datetime(df['Date'])
df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year

monthly_performance = df.groupby('Month').agg({
    'Correct_Prediction': ['count', 'sum', 'mean'],
    'Signal_Strength': 'mean',
    'Confidence': 'mean'
}).round(4)
monthly_performance.columns = ['Count', 'Correct', 'Accuracy', 'Avg_Strength', 'Avg_Confidence']
print("Monthly Performance:")
print(monthly_performance)
print()

# Correlations
print("11. CORRELATION ANALYSIS:")
correlations = df[['Signal_Probability', 'Signal_Strength', 'Confidence', 'Correct_Prediction', 'Actual_Return']].corr()
print("Key correlations:")
print(f"Signal_Probability vs Correct_Prediction: {correlations.loc['Signal_Probability', 'Correct_Prediction']:.4f}")
print(f"Confidence vs Correct_Prediction: {correlations.loc['Confidence', 'Correct_Prediction']:.4f}")
print(f"Signal_Strength vs Correct_Prediction: {correlations.loc['Signal_Strength', 'Correct_Prediction']:.4f}")
print()

# Model comparison
print("12. COMPARISON WITH PREVIOUS RESULTS:")
print("Previous Enhanced Strategy Results:")
print("   - Accuracy: 63.57%")
print("   - Opposite predictions: 36.4%")
print("   - Total signals: 678")
print()
print("Current Optimized Strategy Results:")
print(f"   - Accuracy: {accuracy:.2f}%")
print(f"   - Opposite predictions: {opposite_count/len(df)*100:.1f}%")
print(f"   - Total signals: {len(df)}")
print()

# Performance improvement
previous_accuracy = 63.57
previous_opposite = 36.4
current_accuracy = accuracy
current_opposite = opposite_count/len(df)*100

accuracy_improvement = current_accuracy - previous_accuracy
opposite_improvement = previous_opposite - current_opposite

print("PERFORMANCE IMPROVEMENTS:")
print(f"   Accuracy change: {accuracy_improvement:+.2f} percentage points")
print(f"   Opposite prediction change: {opposite_improvement:+.2f} percentage points")

if accuracy_improvement > 0:
    print(f"   ✅ Accuracy IMPROVED by {accuracy_improvement:.2f}%")
else:
    print(f"   ⚠️ Accuracy DECREASED by {abs(accuracy_improvement):.2f}%")

if opposite_improvement > 0:
    print(f"   ✅ Opposite predictions REDUCED by {opposite_improvement:.2f}%")
else:
    print(f"   ⚠️ Opposite predictions INCREASED by {abs(opposite_improvement):.2f}%")
print()

# Key insights
print("13. KEY INSIGHTS:")
print("✅ Model Performance:")
print(f"   - The optimized model shows {accuracy:.1f}% accuracy")
print(f"   - {opposite_count/len(df)*100:.1f}% opposite predictions")
print(f"   - Generated {len(df)} signals from synthetic data")
print()

print("✅ Signal Quality:")
high_prob_signals = (df['Signal_Probability'] > 0.65).sum()
print(f"   - {high_prob_signals} signals ({high_prob_signals/len(df)*100:.1f}%) with probability > 0.65")
print(f"   - Average confidence: {df['Confidence'].mean():.3f}")
print(f"   - All signals are bullish (model learned trend-following behavior)")
print()

print("✅ Model Characteristics:")
print(f"   - Strong bias toward bullish predictions ({signal_counts.get(1, 0)/len(df)*100:.1f}%)")
print(f"   - High confidence in predictions (avg: {df['Confidence'].mean():.3f})")
print(f"   - Consistent signal strength distribution")
print()

# Recommendations
print("14. RECOMMENDATIONS FOR FURTHER IMPROVEMENT:")
print("🎯 Model Balance:")
print("   - Address bullish bias by balancing training data")
print("   - Implement class weighting to handle imbalanced signals")
print("   - Add market regime detection for adaptive thresholds")
print()

print("🎯 Signal Quality:")
print("   - Implement ensemble methods with multiple models")
print("   - Add volatility-based confidence adjustments")
print("   - Use time-series cross-validation for better generalization")
print()

print("🎯 Real Data Testing:")
print("   - Test with actual market data instead of synthetic")
print("   - Validate across multiple market conditions")
print("   - Implement out-of-sample testing periods")
print()

# Save summary
summary_text = f"""
=== OPTIMIZED LSTM STRATEGY ANALYSIS SUMMARY ===
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

PERFORMANCE METRICS:
- Total Signals: {len(df)}
- Overall Accuracy: {accuracy:.2f}%
- Opposite Predictions: {opposite_count/len(df)*100:.1f}%
- Average Confidence: {df['Confidence'].mean():.4f}
- Average Signal Strength: {df['Signal_Strength'].mean():.4f}

SIGNAL DISTRIBUTION:
- Bullish Signals: {signal_counts.get(1, 0)} ({signal_counts.get(1, 0)/len(df)*100:.1f}%)
- Bearish Signals: {signal_counts.get(0, 0)} ({signal_counts.get(0, 0)/len(df)*100:.1f}%)

COMPARISON WITH PREVIOUS RESULTS:
- Previous Accuracy: 63.57% → Current: {accuracy:.2f}% ({accuracy_improvement:+.2f}%)
- Previous Opposite: 36.4% → Current: {opposite_count/len(df)*100:.1f}% ({opposite_improvement:+.2f}%)

KEY FINDINGS:
- Model shows strong bullish bias in synthetic data
- High confidence levels across all predictions
- Need for class balancing and real data validation
"""

with open('output/optimized_strategy_analysis_summary.txt', 'w') as f:
    f.write(summary_text)

print("✅ Analysis complete! Summary saved to output/optimized_strategy_analysis_summary.txt")
