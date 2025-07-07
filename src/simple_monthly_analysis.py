import csv
from collections import defaultdict
from datetime import datetime

def analyze_monthly_performance_simple():
    """
    Simple monthly analysis without pandas
    """
    
    # Read the CSV file
    signals = []
    with open('output/improved_signals_filtered.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            signals.append(row)
    
    print("=== DETAILED MONTHLY ANALYSIS: COMPREHENSIVE STRATEGY ===")
    print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total Signals: {len(signals)}")
    
    # Calculate overall accuracy
    correct_predictions = sum(1 for s in signals if s['Correct_Prediction'] == '1')
    overall_accuracy = correct_predictions / len(signals)
    print(f"Overall Accuracy: {overall_accuracy:.2%}")
    
    # Group by month
    monthly_data = defaultdict(lambda: {
        'total': 0,
        'correct': 0,
        'bullish_signals': 0,
        'bearish_signals': 0,
        'bullish_correct': 0,
        'bearish_correct': 0,
        'actual_bullish': 0,
        'actual_bearish': 0,
        'total_probability': 0,
        'total_strength': 0
    })
    
    for signal in signals:
        month = int(signal['Month'])
        monthly_data[month]['total'] += 1
        monthly_data[month]['total_probability'] += float(signal['Signal_Probability'])
        monthly_data[month]['total_strength'] += float(signal['Signal_Strength'])
        
        if signal['Correct_Prediction'] == '1':
            monthly_data[month]['correct'] += 1
        
        if signal['Signal_Direction'] == 'Bullish':
            monthly_data[month]['bullish_signals'] += 1
            if signal['Correct_Prediction'] == '1':
                monthly_data[month]['bullish_correct'] += 1
        else:
            monthly_data[month]['bearish_signals'] += 1
            if signal['Correct_Prediction'] == '1':
                monthly_data[month]['bearish_correct'] += 1
        
        if signal['Actual_Direction'] == 'Bullish':
            monthly_data[month]['actual_bullish'] += 1
        else:
            monthly_data[month]['actual_bearish'] += 1
    
    print("\n=== MONTHLY BREAKDOWN ===")
    print("Month | Total | Correct | Accuracy | Bullish | Bearish | Bull_Acc | Bear_Acc | Market_Tendency")
    print("-" * 95)
    
    for month in sorted(monthly_data.keys()):
        data = monthly_data[month]
        accuracy = data['correct'] / data['total'] if data['total'] > 0 else 0
        bullish_acc = data['bullish_correct'] / data['bullish_signals'] if data['bullish_signals'] > 0 else 0
        bearish_acc = data['bearish_correct'] / data['bearish_signals'] if data['bearish_signals'] > 0 else 0
        market_tendency = data['actual_bullish'] / data['total'] if data['total'] > 0 else 0
        
        print(f"{month:5d} | {data['total']:5d} | {data['correct']:7d} | {accuracy:8.1%} | "
              f"{data['bullish_signals']:7d} | {data['bearish_signals']:7d} | {bullish_acc:8.1%} | "
              f"{bearish_acc:8.1%} | {market_tendency:8.1%}")
    
    print("\n=== BULLISH SIGNALS BY MONTH ===")
    print("Month | Bullish_Signals | Correct_Bullish | Accuracy")
    print("-" * 50)
    
    for month in sorted(monthly_data.keys()):
        data = monthly_data[month]
        bullish_acc = data['bullish_correct'] / data['bullish_signals'] if data['bullish_signals'] > 0 else 0
        print(f"{month:5d} | {data['bullish_signals']:15d} | {data['bullish_correct']:15d} | {bullish_acc:8.1%}")
    
    print("\n=== BEARISH SIGNALS BY MONTH ===")
    print("Month | Bearish_Signals | Correct_Bearish | Accuracy")
    print("-" * 50)
    
    for month in sorted(monthly_data.keys()):
        data = monthly_data[month]
        bearish_acc = data['bearish_correct'] / data['bearish_signals'] if data['bearish_signals'] > 0 else 0
        print(f"{month:5d} | {data['bearish_signals']:15d} | {data['bearish_correct']:15d} | {bearish_acc:8.1%}")
    
    # Company analysis
    company_data = defaultdict(lambda: {
        'total': 0,
        'correct': 0,
        'total_probability': 0,
        'total_strength': 0
    })
    
    for signal in signals:
        company = signal['Company']
        company_data[company]['total'] += 1
        company_data[company]['total_probability'] += float(signal['Signal_Probability'])
        company_data[company]['total_strength'] += float(signal['Signal_Strength'])
        
        if signal['Correct_Prediction'] == '1':
            company_data[company]['correct'] += 1
    
    print("\n=== COMPANY PERFORMANCE ===")
    print("Company | Total | Correct | Accuracy | Avg_Probability | Avg_Strength")
    print("-" * 70)
    
    for company in sorted(company_data.keys()):
        data = company_data[company]
        accuracy = data['correct'] / data['total'] if data['total'] > 0 else 0
        avg_prob = data['total_probability'] / data['total'] if data['total'] > 0 else 0
        avg_strength = data['total_strength'] / data['total'] if data['total'] > 0 else 0
        
        print(f"{company:7s} | {data['total']:5d} | {data['correct']:7d} | {accuracy:8.1%} | "
              f"{avg_prob:15.4f} | {avg_strength:12.4f}")
    
    # Calculate holding period
    company_dates = defaultdict(list)
    for signal in signals:
        company_dates[signal['Company']].append(signal['Date'])
    
    holding_periods = []
    for company, dates in company_dates.items():
        sorted_dates = sorted(dates)
        for i in range(1, len(sorted_dates)):
            try:
                date1 = datetime.strptime(sorted_dates[i-1], '%Y-%m-%d')
                date2 = datetime.strptime(sorted_dates[i], '%Y-%m-%d')
                diff = (date2 - date1).days
                holding_periods.append(diff)
            except:
                continue
    
    if holding_periods:
        avg_holding = sum(holding_periods) / len(holding_periods)
        print(f"\n=== HOLDING PERIOD ANALYSIS ===")
        print(f"Average Holding Period: {avg_holding:.1f} days")
        print(f"Median Holding Period: {sorted(holding_periods)[len(holding_periods)//2]:.1f} days")
        print(f"Min Holding Period: {min(holding_periods):.1f} days")
        print(f"Max Holding Period: {max(holding_periods):.1f} days")
    
    # Market conditions
    print("\n=== MARKET CONDITION ANALYSIS ===")
    print("Month | Bullish_Market_Tendency | Our_Performance_in_Market")
    print("-" * 55)
    
    for month in sorted(monthly_data.keys()):
        data = monthly_data[month]
        market_tendency = data['actual_bullish'] / data['total'] if data['total'] > 0 else 0
        our_accuracy = data['correct'] / data['total'] if data['total'] > 0 else 0
        market_type = "Bullish" if market_tendency > 0.5 else "Bearish"
        
        print(f"{month:5d} | {market_tendency:23.1%} | {our_accuracy:15.1%} ({market_type})")
    
    # Returns analysis (simplified)
    print("\n=== RETURN ANALYSIS ===")
    total_return = 0
    monthly_returns = defaultdict(float)
    
    for signal in signals:
        month = int(signal['Month'])
        price_change = float(signal['Close_Price'])
        
        if signal['Correct_Prediction'] == '1':
            # Correct prediction - we gain the price movement
            total_return += abs(price_change)
            monthly_returns[month] += abs(price_change)
        else:
            # Incorrect prediction - we lose the price movement
            total_return -= abs(price_change)
            monthly_returns[month] -= abs(price_change)
    
    print(f"Total Theoretical Return: {total_return:.4f}")
    print(f"Average Return per Signal: {total_return/len(signals):.4f}")
    
    print("\n=== MONTHLY RETURNS ===")
    for month in sorted(monthly_returns.keys()):
        print(f"Month {month}: {monthly_returns[month]:.4f}")
    
    # Save results
    with open('output/detailed_monthly_analysis_simple.txt', 'w') as f:
        f.write("=== DETAILED MONTHLY ANALYSIS: COMPREHENSIVE STRATEGY ===\n")
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Signals: {len(signals)}\n")
        f.write(f"Overall Accuracy: {overall_accuracy:.2%}\n\n")
        
        f.write("=== MONTHLY BREAKDOWN ===\n")
        f.write("Month | Total | Correct | Accuracy | Bullish | Bearish | Bull_Acc | Bear_Acc | Market_Tendency\n")
        f.write("-" * 95 + "\n")
        
        for month in sorted(monthly_data.keys()):
            data = monthly_data[month]
            accuracy = data['correct'] / data['total'] if data['total'] > 0 else 0
            bullish_acc = data['bullish_correct'] / data['bullish_signals'] if data['bullish_signals'] > 0 else 0
            bearish_acc = data['bearish_correct'] / data['bearish_signals'] if data['bearish_signals'] > 0 else 0
            market_tendency = data['actual_bullish'] / data['total'] if data['total'] > 0 else 0
            
            f.write(f"{month:5d} | {data['total']:5d} | {data['correct']:7d} | {accuracy:8.1%} | "
                   f"{data['bullish_signals']:7d} | {data['bearish_signals']:7d} | {bullish_acc:8.1%} | "
                   f"{bearish_acc:8.1%} | {market_tendency:8.1%}\n")
        
        f.write(f"\n=== HOLDING PERIOD ANALYSIS ===\n")
        if holding_periods:
            f.write(f"Average Holding Period: {avg_holding:.1f} days\n")
            f.write(f"Median Holding Period: {sorted(holding_periods)[len(holding_periods)//2]:.1f} days\n")
        
        f.write(f"\n=== RETURN ANALYSIS ===\n")
        f.write(f"Total Theoretical Return: {total_return:.4f}\n")
        f.write(f"Average Return per Signal: {total_return/len(signals):.4f}\n")
        
        f.write("\n=== MONTHLY RETURNS ===\n")
        for month in sorted(monthly_returns.keys()):
            f.write(f"Month {month}: {monthly_returns[month]:.4f}\n")
    
    print("\nAnalysis complete! Results saved to 'output/detailed_monthly_analysis_simple.txt'")

if __name__ == "__main__":
    analyze_monthly_performance_simple()
