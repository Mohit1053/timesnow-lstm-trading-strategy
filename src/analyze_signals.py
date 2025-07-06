import pandas as pd
import numpy as np

# Load the signals data
df = pd.read_csv('output/rolling_window_signals_20250706_114025.csv')

print('🔍 COMPREHENSIVE ANALYSIS OF ROLLING WINDOW LSTM SIGNALS')
print('='*60)

# Basic statistics
print(f'📊 DATASET OVERVIEW:')
print(f'   Total signals: {len(df):,}')
print(f'   Date range: {df["Date"].min()} to {df["Date"].max()}')
print(f'   Companies: {df["Company"].nunique()} ({list(df["Company"].unique())})')
print()

# Signal distribution
print(f'📈 SIGNAL DISTRIBUTION:')
signal_counts = df['Signal_Direction'].value_counts()
print(f'   Bullish signals: {signal_counts.get("Bullish", 0):,} ({signal_counts.get("Bullish", 0)/len(df)*100:.1f}%)')
print(f'   Bearish signals: {signal_counts.get("Bearish", 0):,} ({signal_counts.get("Bearish", 0)/len(df)*100:.1f}%)')
print()

# Accuracy analysis
print(f'🎯 ACCURACY ANALYSIS:')
overall_accuracy = df['Correct_Prediction'].mean()
print(f'   Overall accuracy: {overall_accuracy:.2%}')

# Accuracy by signal type
bullish_acc = df[df['Signal_Direction'] == 'Bullish']['Correct_Prediction'].mean()
bearish_acc = df[df['Signal_Direction'] == 'Bearish']['Correct_Prediction'].mean()
print(f'   Bullish accuracy: {bullish_acc:.2%}')
print(f'   Bearish accuracy: {bearish_acc:.2%}')
print()

# Company-wise performance
print(f'🏢 COMPANY-WISE PERFORMANCE:')
company_stats = df.groupby('Company').agg({
    'Correct_Prediction': ['mean', 'count'],
    'Signal_Strength': 'mean'
}).round(3)
company_stats.columns = ['Accuracy', 'Total_Signals', 'Avg_Signal_Strength']
print(company_stats)
print()

# Signal strength analysis
print(f'💪 SIGNAL STRENGTH ANALYSIS:')
print(f'   Average signal strength: {df["Signal_Strength"].mean():.3f}')
print(f'   Signal strength std: {df["Signal_Strength"].std():.3f}')
print(f'   Strong signals (>0.5): {(df["Signal_Strength"] > 0.5).sum():,} ({(df["Signal_Strength"] > 0.5).mean()*100:.1f}%)')
print(f'   Weak signals (<0.2): {(df["Signal_Strength"] < 0.2).sum():,} ({(df["Signal_Strength"] < 0.2).mean()*100:.1f}%)')
print()

# Warning analysis
print(f'⚠️ WARNING ANALYSIS:')
opposite_predictions = (df['Warning'] == 'OPPOSITE_PREDICTION').sum()
print(f'   Opposite predictions: {opposite_predictions:,} ({opposite_predictions/len(df)*100:.1f}%)')
print(f'   Correct predictions: {(df["Warning"] == "CORRECT_PREDICTION").sum():,}')
print()

# Monthly performance
print(f'📅 MONTHLY PERFORMANCE:')
df['Date'] = pd.to_datetime(df['Date'])
df['Month'] = df['Date'].dt.to_period('M')
monthly_acc = df.groupby('Month')['Correct_Prediction'].agg(['mean', 'count']).round(3)
monthly_acc.columns = ['Accuracy', 'Signals']
print(monthly_acc.head(10))
print()

# Best and worst performing periods
print(f'🏆 BEST PERFORMING MONTHS:')
best_months = monthly_acc.nlargest(5, 'Accuracy')
print(best_months)
print()

print(f'📉 WORST PERFORMING MONTHS:')
worst_months = monthly_acc.nsmallest(5, 'Accuracy')
print(worst_months)
print()

# Signal strength vs accuracy correlation
print(f'📊 SIGNAL STRENGTH vs ACCURACY:')
strong_signals = df[df['Signal_Strength'] > 0.5]
weak_signals = df[df['Signal_Strength'] < 0.2]
print(f'   Strong signals accuracy: {strong_signals["Correct_Prediction"].mean():.2%}')
print(f'   Weak signals accuracy: {weak_signals["Correct_Prediction"].mean():.2%}')
print()

# Time series analysis
print(f'📈 TIME SERIES TRENDS:')
df['Week'] = df['Date'].dt.isocalendar().week
df['Quarter'] = df['Date'].dt.quarter
df['Year'] = df['Date'].dt.year

yearly_acc = df.groupby('Year')['Correct_Prediction'].agg(['mean', 'count']).round(3)
yearly_acc.columns = ['Accuracy', 'Signals']
print('Year-wise performance:')
print(yearly_acc)
print()

# Consecutive performance
print(f'🔄 CONSECUTIVE PERFORMANCE ANALYSIS:')
df_sorted = df.sort_values('Date')
df_sorted['Consecutive'] = (df_sorted['Correct_Prediction'] == df_sorted['Correct_Prediction'].shift()).cumsum()
consecutive_streaks = df_sorted.groupby('Consecutive').size()
print(f'   Longest winning streak: {consecutive_streaks.max()}')
print(f'   Average streak length: {consecutive_streaks.mean():.1f}')
print()

# Final summary
print('📝 EXECUTIVE SUMMARY:')
print(f'   • Model achieved {overall_accuracy:.1%} overall accuracy on 5-day predictions')
print(f'   • Balanced performance: Bullish ({bullish_acc:.1%}) vs Bearish ({bearish_acc:.1%})')
print(f'   • Generated {len(df):,} signals across {df["Company"].nunique()} companies')
print(f'   • {opposite_predictions:,} opposite predictions require attention')
print(f'   • Signal strength averages {df["Signal_Strength"].mean():.2f} with {(df["Signal_Strength"] > 0.5).mean()*100:.1f}% strong signals')
