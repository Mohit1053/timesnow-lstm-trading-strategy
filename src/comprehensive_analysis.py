import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Read the improved signals data
df = pd.read_csv('output/improved_signals_filtered.csv')

print("=== COMPREHENSIVE ANALYSIS OF IMPROVED SIGNALS ===\n")

# Basic statistics
print("1. BASIC STATISTICS:")
print(f"Total signals: {len(df)}")
print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
print(f"Companies: {df['Company'].unique()}")
print(f"Unique dates: {df['Date'].nunique()}")
print()

# Overall accuracy
correct_predictions = (df['Correct_Prediction'] == 1).sum()
total_predictions = len(df)
accuracy = correct_predictions / total_predictions * 100

print("2. OVERALL ACCURACY:")
print(f"Correct predictions: {correct_predictions}")
print(f"Total predictions: {total_predictions}")
print(f"Overall accuracy: {accuracy:.2f}%")
print()

# Signal distribution
print("3. SIGNAL DISTRIBUTION:")
signal_counts = df['Predicted_Signal'].value_counts()
print(f"Bullish signals (1): {signal_counts.get(1, 0)} ({signal_counts.get(1, 0)/len(df)*100:.1f}%)")
print(f"Bearish signals (0): {signal_counts.get(0, 0)} ({signal_counts.get(0, 0)/len(df)*100:.1f}%)")
print()

# Warning analysis
print("4. WARNING ANALYSIS:")
warning_counts = df['Warning'].value_counts()
opposite_count = warning_counts.get('OPPOSITE_PREDICTION', 0)
correct_count = warning_counts.get('CORRECT_PREDICTION', 0)

print(f"Correct predictions: {correct_count} ({correct_count/len(df)*100:.1f}%)")
print(f"Opposite predictions: {opposite_count} ({opposite_count/len(df)*100:.1f}%)")
print(f"Improvement from original 42.3%: {42.3 - (opposite_count/len(df)*100):.1f} percentage points")
print()

# Company-wise analysis
print("5. COMPANY-WISE ANALYSIS:")
company_stats = df.groupby('Company').agg({
    'Correct_Prediction': ['count', 'sum', 'mean'],
    'Signal_Probability': ['mean', 'std'],
    'Signal_Strength': ['mean', 'std']
}).round(4)

company_stats.columns = ['Total_Signals', 'Correct_Count', 'Accuracy', 
                        'Avg_Probability', 'Std_Probability', 
                        'Avg_Strength', 'Std_Strength']

print(company_stats)
print()

# Monthly performance
print("6. MONTHLY PERFORMANCE:")
monthly_stats = df.groupby('Month').agg({
    'Correct_Prediction': ['count', 'sum', 'mean'],
    'Signal_Probability': 'mean',
    'Signal_Strength': 'mean'
}).round(4)

monthly_stats.columns = ['Total_Signals', 'Correct_Count', 'Accuracy', 'Avg_Probability', 'Avg_Strength']
print(monthly_stats)
print()

# Signal strength analysis
print("7. SIGNAL STRENGTH ANALYSIS:")
print(f"Average signal strength: {df['Signal_Strength'].mean():.4f}")
print(f"Signal strength std: {df['Signal_Strength'].std():.4f}")
print(f"Min signal strength: {df['Signal_Strength'].min():.4f}")
print(f"Max signal strength: {df['Signal_Strength'].max():.4f}")
print()

# High confidence signals
print("8. HIGH CONFIDENCE SIGNALS:")
high_conf_threshold = 0.6
high_conf_signals = df[df['Signal_Probability'] >= high_conf_threshold]
if len(high_conf_signals) > 0:
    high_conf_accuracy = high_conf_signals['Correct_Prediction'].mean() * 100
    print(f"High confidence signals (>= {high_conf_threshold}): {len(high_conf_signals)}")
    print(f"High confidence accuracy: {high_conf_accuracy:.2f}%")
else:
    print("No high confidence signals found")
print()

# Strong signals analysis
print("9. STRONG SIGNALS ANALYSIS:")
strong_threshold = 0.3
strong_signals = df[df['Signal_Strength'] >= strong_threshold]
if len(strong_signals) > 0:
    strong_accuracy = strong_signals['Correct_Prediction'].mean() * 100
    print(f"Strong signals (strength >= {strong_threshold}): {len(strong_signals)}")
    print(f"Strong signals accuracy: {strong_accuracy:.2f}%")
else:
    print("No strong signals found")
print()

# Correlation analysis
print("10. CORRELATION ANALYSIS:")
correlations = df[['Signal_Probability', 'Signal_Strength', 'Correct_Prediction']].corr()
print("Correlation between Signal_Probability and Correct_Prediction:", 
      correlations.loc['Signal_Probability', 'Correct_Prediction'])
print("Correlation between Signal_Strength and Correct_Prediction:", 
      correlations.loc['Signal_Strength', 'Correct_Prediction'])
print("Correlation between Signal_Probability and Signal_Strength:", 
      correlations.loc['Signal_Probability', 'Signal_Strength'])
print()

# Time series analysis
print("11. TIME SERIES PERFORMANCE:")
df['Date'] = pd.to_datetime(df['Date'])
df['Year'] = df['Date'].dt.year
yearly_stats = df.groupby('Year').agg({
    'Correct_Prediction': ['count', 'sum', 'mean'],
    'Signal_Probability': 'mean',
    'Signal_Strength': 'mean'
}).round(4)

yearly_stats.columns = ['Total_Signals', 'Correct_Count', 'Accuracy', 'Avg_Probability', 'Avg_Strength']
print(yearly_stats)
print()

# Best and worst performing periods
print("12. BEST/WORST PERFORMING PERIODS:")
monthly_accuracy = df.groupby('Month')['Correct_Prediction'].mean()
best_month = monthly_accuracy.idxmax()
worst_month = monthly_accuracy.idxmin()
print(f"Best performing month: {best_month} ({monthly_accuracy[best_month]*100:.1f}% accuracy)")
print(f"Worst performing month: {worst_month} ({monthly_accuracy[worst_month]*100:.1f}% accuracy)")
print()

# Signal direction analysis
print("13. SIGNAL DIRECTION ANALYSIS:")
direction_stats = df.groupby(['Signal_Direction', 'Actual_Direction']).size().unstack(fill_value=0)
print("Signal vs Actual Direction Cross-tabulation:")
print(direction_stats)
print()

# Create visualizations
plt.figure(figsize=(16, 12))

# Plot 1: Monthly accuracy
plt.subplot(2, 3, 1)
monthly_accuracy.plot(kind='bar', color='skyblue')
plt.title('Monthly Accuracy')
plt.xlabel('Month')
plt.ylabel('Accuracy')
plt.xticks(rotation=45)

# Plot 2: Company-wise accuracy
plt.subplot(2, 3, 2)
company_accuracy = df.groupby('Company')['Correct_Prediction'].mean()
company_accuracy.plot(kind='bar', color='lightgreen')
plt.title('Company-wise Accuracy')
plt.xlabel('Company')
plt.ylabel('Accuracy')
plt.xticks(rotation=45)

# Plot 3: Signal strength distribution
plt.subplot(2, 3, 3)
plt.hist(df['Signal_Strength'], bins=20, color='orange', alpha=0.7)
plt.title('Signal Strength Distribution')
plt.xlabel('Signal Strength')
plt.ylabel('Frequency')

# Plot 4: Signal probability distribution
plt.subplot(2, 3, 4)
plt.hist(df['Signal_Probability'], bins=20, color='purple', alpha=0.7)
plt.title('Signal Probability Distribution')
plt.xlabel('Signal Probability')
plt.ylabel('Frequency')

# Plot 5: Accuracy over time
plt.subplot(2, 3, 5)
df_sorted = df.sort_values('Date')
df_sorted['Rolling_Accuracy'] = df_sorted['Correct_Prediction'].rolling(window=50).mean()
plt.plot(df_sorted['Date'], df_sorted['Rolling_Accuracy'], color='red')
plt.title('Accuracy Over Time (50-period rolling)')
plt.xlabel('Date')
plt.ylabel('Rolling Accuracy')
plt.xticks(rotation=45)

# Plot 6: Signal strength vs accuracy
plt.subplot(2, 3, 6)
strength_bins = pd.cut(df['Signal_Strength'], bins=10)
strength_accuracy = df.groupby(strength_bins)['Correct_Prediction'].mean()
strength_accuracy.plot(kind='line', marker='o', color='navy')
plt.title('Signal Strength vs Accuracy')
plt.xlabel('Signal Strength Bins')
plt.ylabel('Accuracy')
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('output/comprehensive_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Save summary to file
with open('output/comprehensive_analysis_summary.txt', 'w') as f:
    f.write("=== COMPREHENSIVE ANALYSIS SUMMARY ===\n\n")
    f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Total Signals: {len(df)}\n")
    f.write(f"Overall Accuracy: {accuracy:.2f}%\n")
    f.write(f"Opposite Predictions: {opposite_count} ({opposite_count/len(df)*100:.1f}%)\n")
    f.write(f"Improvement from original 42.3%: {42.3 - (opposite_count/len(df)*100):.1f} percentage points\n\n")
    
    f.write("COMPANY PERFORMANCE:\n")
    for company in df['Company'].unique():
        company_data = df[df['Company'] == company]
        company_accuracy = company_data['Correct_Prediction'].mean() * 100
        f.write(f"{company}: {company_accuracy:.1f}% accuracy ({len(company_data)} signals)\n")
    
    f.write(f"\nBest Month: {best_month} ({monthly_accuracy[best_month]*100:.1f}%)\n")
    f.write(f"Worst Month: {worst_month} ({monthly_accuracy[worst_month]*100:.1f}%)\n")
    
    f.write(f"\nAverage Signal Strength: {df['Signal_Strength'].mean():.4f}\n")
    f.write(f"Average Signal Probability: {df['Signal_Probability'].mean():.4f}\n")

print("Analysis complete! Check output/comprehensive_analysis.png and output/comprehensive_analysis_summary.txt")
