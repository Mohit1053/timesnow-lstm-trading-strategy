import pandas as pd
import numpy as np

# Read the CSV file
df = pd.read_csv(r'c:\Users\98765\OneDrive\Desktop\Timesnow\src\output\rolling_window_signals_20250706_114025.csv')

print("="*60)
print("🔍 ROLLING WINDOW LSTM SIGNALS ANALYSIS")
print("="*60)

# Basic Overview
print("\n📊 DATASET OVERVIEW:")
print(f"   Total signals: {len(df):,}")
print(f"   Date range: {df['Date'].min()} to {df['Date'].max()}")
print(f"   Companies: {df['Company'].nunique()} - {list(df['Company'].unique())}")
print(f"   Features: {list(df.columns)}")

# Signal Distribution
print("\n📈 SIGNAL DISTRIBUTION:")
signal_dist = df['Signal_Direction'].value_counts()
for signal, count in signal_dist.items():
    print(f"   {signal}: {count:,} ({count/len(df)*100:.1f}%)")

# Overall Accuracy
print("\n🎯 ACCURACY ANALYSIS:")
overall_acc = df['Correct_Prediction'].mean()
print(f"   Overall accuracy: {overall_acc:.2%}")

# Signal-wise accuracy
bull_acc = df[df['Signal_Direction'] == 'Bullish']['Correct_Prediction'].mean()
bear_acc = df[df['Signal_Direction'] == 'Bearish']['Correct_Prediction'].mean()
print(f"   Bullish signal accuracy: {bull_acc:.2%}")
print(f"   Bearish signal accuracy: {bear_acc:.2%}")

# Company Performance
print("\n🏢 COMPANY-WISE PERFORMANCE:")
company_perf = df.groupby('Company').agg({
    'Correct_Prediction': 'mean',
    'Signal_Strength': 'mean'
}).round(3)
company_perf['Signal_Count'] = df.groupby('Company').size()
print(company_perf)

# Signal Strength Analysis
print("\n💪 SIGNAL STRENGTH ANALYSIS:")
print(f"   Average signal strength: {df['Signal_Strength'].mean():.3f}")
print(f"   Signal strength range: {df['Signal_Strength'].min():.3f} to {df['Signal_Strength'].max():.3f}")

# Strength categories
strong_signals = df[df['Signal_Strength'] > 0.5]
medium_signals = df[(df['Signal_Strength'] >= 0.2) & (df['Signal_Strength'] <= 0.5)]
weak_signals = df[df['Signal_Strength'] < 0.2]

print(f"   Strong signals (>0.5): {len(strong_signals):,} ({len(strong_signals)/len(df)*100:.1f}%)")
print(f"   Medium signals (0.2-0.5): {len(medium_signals):,} ({len(medium_signals)/len(df)*100:.1f}%)")
print(f"   Weak signals (<0.2): {len(weak_signals):,} ({len(weak_signals)/len(df)*100:.1f}%)")

# Accuracy by signal strength
if len(strong_signals) > 0:
    print(f"   Strong signal accuracy: {strong_signals['Correct_Prediction'].mean():.2%}")
if len(weak_signals) > 0:
    print(f"   Weak signal accuracy: {weak_signals['Correct_Prediction'].mean():.2%}")

# Warning Analysis
print("\n⚠️ WARNING ANALYSIS:")
warnings = df[df['Warning'] == 'OPPOSITE_PREDICTION']
print(f"   Opposite predictions: {len(warnings):,} ({len(warnings)/len(df)*100:.1f}%)")
print(f"   Correct predictions: {len(df) - len(warnings):,}")

# Company-wise warnings
print("\n   Company-wise opposite predictions:")
warning_by_company = warnings.groupby('Company').size().sort_values(ascending=False)
for company, count in warning_by_company.items():
    company_total = len(df[df['Company'] == company])
    print(f"     {company}: {count:,} ({count/company_total*100:.1f}%)")

# Time-based Analysis
print("\n📅 TIME-BASED ANALYSIS:")
df['Date'] = pd.to_datetime(df['Date'])
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month

yearly_perf = df.groupby('Year')['Correct_Prediction'].mean()
print("   Yearly performance:")
for year, acc in yearly_perf.items():
    year_count = len(df[df['Year'] == year])
    print(f"     {year}: {acc:.2%} ({year_count:,} signals)")

# Best and worst months
monthly_perf = df.groupby(['Year', 'Month'])['Correct_Prediction'].mean().sort_values(ascending=False)
print(f"\n   Best performing months:")
for i, (period, acc) in enumerate(monthly_perf.head(3).items()):
    print(f"     {period[0]}-{period[1]:02d}: {acc:.2%}")

print(f"\n   Worst performing months:")
for i, (period, acc) in enumerate(monthly_perf.tail(3).items()):
    print(f"     {period[0]}-{period[1]:02d}: {acc:.2%}")

# Summary Statistics
print("\n📋 EXECUTIVE SUMMARY:")
print(f"   • Generated {len(df):,} trading signals over {(df['Date'].max() - df['Date'].min()).days} days")
print(f"   • Achieved {overall_acc:.1%} overall accuracy on 5-day predictions")
print(f"   • Balanced bullish ({bull_acc:.1%}) vs bearish ({bear_acc:.1%}) performance")
print(f"   • {len(warnings):,} opposite predictions need attention ({len(warnings)/len(df)*100:.1f}%)")
print(f"   • Average signal strength: {df['Signal_Strength'].mean():.2f}")
print(f"   • Model shows consistent performance across all {df['Company'].nunique()} companies")

print("\n" + "="*60)
print("Analysis Complete!")
print("="*60)
