import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_monthly_performance():
    """
    Detailed monthly analysis of the best performing strategy (Comprehensive Analysis)
    """
    
    # Load the comprehensive analysis results
    df = pd.read_csv('output/improved_signals_filtered.csv')
    
    # Convert Date to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Create year-month column for easier grouping
    df['Year_Month'] = df['Date'].dt.to_period('M')
    
    print("=== DETAILED MONTHLY ANALYSIS: COMPREHENSIVE STRATEGY ===")
    print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total Signals: {len(df)}")
    print(f"Overall Accuracy: {df['Correct_Prediction'].mean():.2%}")
    print(f"Date Range: {df['Date'].min()} to {df['Date'].max()}")
    print()
    
    # Monthly signal counts and accuracy
    monthly_stats = df.groupby('Month').agg({
        'Correct_Prediction': ['count', 'sum', 'mean'],
        'Signal_Direction': lambda x: (x == 'Bullish').sum(),
        'Actual_Direction': lambda x: (x == 'Bullish').sum(),
        'Signal_Probability': 'mean',
        'Signal_Strength': 'mean'
    }).round(4)
    
    monthly_stats.columns = ['Total_Signals', 'Correct_Signals', 'Accuracy', 
                           'Bullish_Predictions', 'Actual_Bullish', 'Avg_Probability', 'Avg_Strength']
    
    monthly_stats['Bearish_Predictions'] = monthly_stats['Total_Signals'] - monthly_stats['Bullish_Predictions']
    monthly_stats['Actual_Bearish'] = monthly_stats['Total_Signals'] - monthly_stats['Actual_Bullish']
    
    print("=== MONTHLY BREAKDOWN ===")
    print(monthly_stats.to_string())
    print()
    
    # Bullish signal accuracy by month
    bullish_signals = df[df['Signal_Direction'] == 'Bullish']
    bullish_monthly = bullish_signals.groupby('Month').agg({
        'Correct_Prediction': ['count', 'sum', 'mean']
    }).round(4)
    bullish_monthly.columns = ['Bullish_Signals', 'Correct_Bullish', 'Bullish_Accuracy']
    
    # Bearish signal accuracy by month
    bearish_signals = df[df['Signal_Direction'] == 'Bearish']
    bearish_monthly = bearish_signals.groupby('Month').agg({
        'Correct_Prediction': ['count', 'sum', 'mean']
    }).round(4)
    bearish_monthly.columns = ['Bearish_Signals', 'Correct_Bearish', 'Bearish_Accuracy']
    
    print("=== BULLISH SIGNALS BY MONTH ===")
    print(bullish_monthly.to_string())
    print()
    
    print("=== BEARISH SIGNALS BY MONTH ===")
    print(bearish_monthly.to_string())
    print()
    
    # Calculate holding period and returns analysis
    print("=== HOLDING PERIOD AND RETURNS ANALYSIS ===")
    
    # Group by company and calculate consecutive signal patterns
    company_analysis = df.groupby('Company').agg({
        'Correct_Prediction': ['count', 'sum', 'mean'],
        'Signal_Probability': 'mean',
        'Signal_Strength': 'mean',
        'Close_Price': 'mean'
    }).round(4)
    
    company_analysis.columns = ['Total_Signals', 'Correct_Signals', 'Accuracy', 
                               'Avg_Probability', 'Avg_Strength', 'Avg_Price_Change']
    
    print("=== COMPANY PERFORMANCE ===")
    print(company_analysis.to_string())
    print()
    
    # Calculate average holding period (assuming signals are held until next signal)
    holding_periods = []
    for company in df['Company'].unique():
        company_data = df[df['Company'] == company].sort_values('Date')
        if len(company_data) > 1:
            time_diffs = company_data['Date'].diff().dt.days.dropna()
            holding_periods.extend(time_diffs.tolist())
    
    avg_holding_period = np.mean(holding_periods) if holding_periods else 0
    
    print(f"=== AVERAGE HOLDING PERIOD ===")
    print(f"Average Holding Period: {avg_holding_period:.1f} days")
    print(f"Median Holding Period: {np.median(holding_periods):.1f} days")
    print(f"Min Holding Period: {np.min(holding_periods):.1f} days")
    print(f"Max Holding Period: {np.max(holding_periods):.1f} days")
    print()
    
    # Market condition analysis
    print("=== MARKET CONDITION ANALYSIS ===")
    
    # Calculate market condition based on actual direction distribution
    market_bullish_months = df.groupby('Month')['Actual_Direction'].apply(
        lambda x: (x == 'Bullish').sum() / len(x)
    ).round(4)
    
    print("Market Bullish Tendency by Month:")
    for month, bullish_pct in market_bullish_months.items():
        condition = "Bullish Market" if bullish_pct > 0.5 else "Bearish Market"
        print(f"Month {month}: {bullish_pct:.1%} bullish - {condition}")
    print()
    
    # Performance in bullish vs bearish market conditions
    df['Market_Condition'] = df.groupby('Month')['Actual_Direction'].transform(
        lambda x: 'Bullish_Market' if (x == 'Bullish').sum() / len(x) > 0.5 else 'Bearish_Market'
    )
    
    market_performance = df.groupby('Market_Condition').agg({
        'Correct_Prediction': ['count', 'sum', 'mean'],
        'Signal_Probability': 'mean',
        'Signal_Strength': 'mean'
    }).round(4)
    
    market_performance.columns = ['Total_Signals', 'Correct_Signals', 'Accuracy', 
                                'Avg_Probability', 'Avg_Strength']
    
    print("=== PERFORMANCE BY MARKET CONDITION ===")
    print(market_performance.to_string())
    print()
    
    # Calculate approximate returns based on price changes
    print("=== RETURN ANALYSIS ===")
    
    # Assuming Close_Price represents normalized price change
    correct_predictions = df[df['Correct_Prediction'] == 1]
    incorrect_predictions = df[df['Correct_Prediction'] == 0]
    
    # For correct predictions, we would have gained the price movement
    # For incorrect predictions, we would have lost the price movement
    total_return = correct_predictions['Close_Price'].sum() - incorrect_predictions['Close_Price'].abs().sum()
    avg_return_per_signal = total_return / len(df)
    
    print(f"Total Theoretical Return: {total_return:.4f}")
    print(f"Average Return per Signal: {avg_return_per_signal:.4f}")
    print(f"Return from Correct Predictions: {correct_predictions['Close_Price'].sum():.4f}")
    print(f"Loss from Incorrect Predictions: {incorrect_predictions['Close_Price'].abs().sum():.4f}")
    print()
    
    # Monthly returns
    monthly_returns = df.groupby('Month').apply(
        lambda x: x[x['Correct_Prediction'] == 1]['Close_Price'].sum() - 
                  x[x['Correct_Prediction'] == 0]['Close_Price'].abs().sum()
    ).round(4)
    
    print("=== MONTHLY RETURNS ===")
    for month, ret in monthly_returns.items():
        print(f"Month {month}: {ret:.4f}")
    print()
    
    # Save detailed results
    with open('output/detailed_monthly_analysis.txt', 'w') as f:
        f.write("=== DETAILED MONTHLY ANALYSIS: COMPREHENSIVE STRATEGY ===\n")
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Signals: {len(df)}\n")
        f.write(f"Overall Accuracy: {df['Correct_Prediction'].mean():.2%}\n")
        f.write(f"Date Range: {df['Date'].min()} to {df['Date'].max()}\n\n")
        
        f.write("=== MONTHLY BREAKDOWN ===\n")
        f.write(monthly_stats.to_string())
        f.write("\n\n")
        
        f.write("=== BULLISH SIGNALS BY MONTH ===\n")
        f.write(bullish_monthly.to_string())
        f.write("\n\n")
        
        f.write("=== BEARISH SIGNALS BY MONTH ===\n")
        f.write(bearish_monthly.to_string())
        f.write("\n\n")
        
        f.write(f"=== AVERAGE HOLDING PERIOD ===\n")
        f.write(f"Average Holding Period: {avg_holding_period:.1f} days\n")
        f.write(f"Median Holding Period: {np.median(holding_periods):.1f} days\n\n")
        
        f.write("=== PERFORMANCE BY MARKET CONDITION ===\n")
        f.write(market_performance.to_string())
        f.write("\n\n")
        
        f.write("=== MONTHLY RETURNS ===\n")
        for month, ret in monthly_returns.items():
            f.write(f"Month {month}: {ret:.4f}\n")
    
    print("Analysis complete! Results saved to 'output/detailed_monthly_analysis.txt'")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Monthly accuracy
    monthly_stats['Accuracy'].plot(kind='bar', ax=axes[0,0], color='skyblue')
    axes[0,0].set_title('Monthly Accuracy')
    axes[0,0].set_ylabel('Accuracy %')
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # Monthly signal counts
    monthly_stats[['Bullish_Predictions', 'Bearish_Predictions']].plot(kind='bar', ax=axes[0,1], stacked=True)
    axes[0,1].set_title('Monthly Signal Distribution')
    axes[0,1].set_ylabel('Number of Signals')
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # Monthly returns
    monthly_returns.plot(kind='bar', ax=axes[1,0], color='green')
    axes[1,0].set_title('Monthly Returns')
    axes[1,0].set_ylabel('Return')
    axes[1,0].tick_params(axis='x', rotation=45)
    
    # Company performance
    company_analysis['Accuracy'].plot(kind='bar', ax=axes[1,1], color='orange')
    axes[1,1].set_title('Company Performance')
    axes[1,1].set_ylabel('Accuracy %')
    axes[1,1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('output/detailed_monthly_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Visualization saved to 'output/detailed_monthly_analysis.png'")

if __name__ == "__main__":
    analyze_monthly_performance()
