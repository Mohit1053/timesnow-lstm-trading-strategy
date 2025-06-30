#!/usr/bin/env python3
"""
Backtesting Script for Focused LSTM Trading Strategy

This script performs comprehensive backtesting of the focused LSTM strategy using:
- 13 specific technical indicators
- Walk-forward validation
- Realistic portfolio simulation
- Performance metrics and reporting
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

# Import required modules
from technical_indicators import (
    calculate_rsi, calculate_roc, calculate_stochastic, calculate_tsi,
    calculate_obv, calculate_mfi, calculate_pvt,
    calculate_tema, calculate_macd, calculate_kama,
    calculate_atr, calculate_bollinger_bands, calculate_ulcer_index
)
from lstm_trading_strategy import LSTMTradingStrategy

import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

class FocusedLSTMBacktester:
    """Backtesting framework specifically for the focused LSTM strategy"""
    
    def __init__(self, initial_capital=100, commission=0.001, slippage=0.0005):
        """
        Initialize the backtester
        
        Args:
            initial_capital (float): Starting capital for backtesting
            commission (float): Commission rate per trade (0.001 = 0.1%)
            slippage (float): Slippage rate per trade (0.0005 = 0.05%)
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        
    def load_and_prepare_data(self, file_path):
        """Load price data and calculate focused indicators"""
        print(f"📊 Loading and preparing data from: {file_path}")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        df = pd.read_csv(file_path)
        
        # Check if this is the multi-company dataset format
        if 'PriceDate' in df.columns and 'companyid' in df.columns:
            print(f"📊 Detected multi-company dataset with {df['companyid'].nunique()} companies")
            
            # Select the company with the most data
            company_counts = df['companyid'].value_counts()
            selected_company = company_counts.index[0]
            company_data_count = company_counts.iloc[0]
            
            print(f"🎯 Selected company {selected_company} with {company_data_count} records")
            df = df[df['companyid'] == selected_company].copy()
            
            # Rename columns to standard format
            column_mapping = {
                'PriceDate': 'Date',
                'OpenPrice': 'Open',
                'AdjustedHighPrice': 'High',
                'AdjustedLowPrice': 'Low',
                'AdjustedClosePrice': 'Close',
                'TradedQuantity': 'Volume'
            }
            df = df.rename(columns=column_mapping)
            
            # Keep only required columns
            required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            df = df[required_columns]
        
        # Convert Date column and set as index
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        df.sort_index(inplace=True)
        
        # Calculate focused indicators
        df = self._calculate_focused_indicators(df)
        
        print(f"✅ Prepared {len(df)} records from {df.index[0].date()} to {df.index[-1].date()}")
        return df
    
    def _calculate_focused_indicators(self, df):
        """Calculate the 13 focused technical indicators"""
        print("🔧 Calculating focused technical indicators...")
        
        high = df['High']
        low = df['Low']
        close = df['Close']
        open_price = df['Open']
        volume = df['Volume'] if 'Volume' in df.columns else pd.Series(index=df.index, dtype=float)
        
        # Create result dataframe
        result_df = df.copy()
        
        # 📈 Momentum Indicators
        result_df['RSI'] = calculate_rsi(close)
        result_df['ROC'] = calculate_roc(close)
        
        # Stochastic Oscillator
        stoch_k, stoch_d = calculate_stochastic(high, low, close)
        result_df['Stoch_K'] = stoch_k
        result_df['Stoch_D'] = stoch_d
        
        result_df['TSI'] = calculate_tsi(close)
        
        # 📊 Volume Indicators
        result_df['OBV'] = calculate_obv(close, volume)
        result_df['MFI'] = calculate_mfi(high, low, close, volume)
        result_df['PVT'] = calculate_pvt(close, volume)
        
        # 📉 Trend Indicators
        result_df['TEMA'] = calculate_tema(close)
        
        # MACD
        macd_line, signal_line, histogram = calculate_macd(close)
        result_df['MACD'] = macd_line
        result_df['MACD_Signal'] = signal_line
        result_df['MACD_Histogram'] = histogram
        
        result_df['KAMA'] = calculate_kama(close)
        
        # 🌪️ Volatility Indicators
        result_df['ATR'] = calculate_atr(high, low, close)
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(close)
        result_df['BB_Upper'] = bb_upper
        result_df['BB_Middle'] = bb_middle
        result_df['BB_Lower'] = bb_lower
        result_df['BB_Width'] = bb_upper - bb_lower
        result_df['BB_Position'] = (close - bb_lower) / (bb_upper - bb_lower)
        
        result_df['Ulcer_Index'] = calculate_ulcer_index(close)
        
        return result_df
    
    def walk_forward_validation(self, df, train_periods=252*3, test_periods=63, step_size=21):
        """
        Perform walk-forward validation
        
        Args:
            df (pd.DataFrame): Data with indicators
            train_periods (int): Number of periods for training (default: 3 years)
            test_periods (int): Number of periods for testing (default: 3 months)
            step_size (int): Step size for rolling window (default: 1 month)
        """
        print(f"🔄 Starting walk-forward validation...")
        print(f"   Training periods: {train_periods}")
        print(f"   Testing periods: {test_periods}")
        print(f"   Step size: {step_size}")
        
        results = []
        total_start = train_periods
        total_end = len(df) - test_periods
        
        fold_num = 1
        
        for start in range(total_start, total_end, step_size):
            train_start = start - train_periods
            train_end = start
            test_start = start
            test_end = min(start + test_periods, len(df))
            
            if test_end - test_start < 30:  # Skip if test period too short
                continue
                
            print(f"\n📊 Fold {fold_num}: Training {df.index[train_start].date()} to {df.index[train_end-1].date()}")
            print(f"                Testing {df.index[test_start].date()} to {df.index[test_end-1].date()}")
            
            # Split data
            train_data = df.iloc[train_start:train_end].copy()
            test_data = df.iloc[test_start:test_end].copy()
            
            try:
                # Train focused LSTM strategy
                strategy = LSTMTradingStrategy(
                    sequence_length=60,
                    target_horizon=5,
                    use_advanced_preprocessing=True,
                    outlier_threshold=2.0,
                    pca_variance_threshold=0.98
                )
                
                # Prepare training data
                X_train, y_train, feature_columns, dates = strategy.prepare_data(train_data)
                
                # Build and train model
                strategy.build_model(lstm_units=[100, 50], dropout_rate=0.2, optimize_features=False)
                strategy.train_model(epochs=30, batch_size=32, validation_split=0.2)
                
                # Make predictions
                strategy.make_predictions()
                
                # Generate signals for test period
                full_data_copy = df.copy()
                signals_df = strategy.generate_trading_signals(full_data_copy)
                
                # Filter signals to test period
                test_start_date = df.index[test_start]
                test_end_date = df.index[test_end-1]
                
                # Backtest the signals
                fold_results = self._backtest_signals(test_data, signals_df, fold_num)
                fold_results['fold'] = fold_num
                fold_results['train_start'] = df.index[train_start]
                fold_results['train_end'] = df.index[train_end-1]
                fold_results['test_start'] = df.index[test_start]
                fold_results['test_end'] = df.index[test_end-1]
                
                results.append(fold_results)
                print(f"✅ Fold {fold_num} completed - Return: {fold_results['total_return']:.2%}")
                
            except Exception as e:
                print(f"❌ Fold {fold_num} failed: {str(e)}")
                continue
            
            fold_num += 1
        
        if not results:
            raise ValueError("No successful folds in walk-forward validation")
        
        # Aggregate results
        aggregated_results = self._aggregate_results(results)
        
        print(f"\n✅ Walk-forward validation completed: {len(results)} successful folds")
        return results, aggregated_results
    
    def _backtest_signals(self, price_data, signals_df, fold_num=None):
        """Backtest trading signals with robust price handling"""
        print(f"🔍 Backtesting {len(signals_df)} signals for fold {fold_num}")
        
        # Initialize portfolio tracking
        capital = self.initial_capital
        position = 0
        trades = []
        portfolio_values = []
        daily_returns = []
        
        # Convert price data to numpy arrays for robust access
        prices = price_data['Close'].values.astype(float)
        dates = price_data.index
        
        # Process each signal with robust price handling
        for i, (signal_idx, row) in enumerate(signals_df.iterrows()):
            if i >= len(prices):
                break
                
            try:
                # Get current price and date safely
                current_price = prices[i]
                current_date = dates[i]
                
                # Extract signal type
                signal = 0
                if 'Signal' in row:
                    signal = int(row['Signal']) if pd.notna(row['Signal']) else 0
                elif 'Signal_Type' in row:
                    signal_type = str(row['Signal_Type']).upper()
                    if signal_type == 'BUY':
                        signal = 1
                    elif signal_type == 'SELL':
                        signal = -1
                
                # Calculate portfolio value
                portfolio_value = capital + (position * current_price)
                portfolio_values.append(portfolio_value)
                
                # Calculate daily return
                if i > 0:
                    daily_return = (portfolio_value - portfolio_values[i-1]) / portfolio_values[i-1]
                    daily_returns.append(float(daily_return))
                
                # Execute trades
                if signal == 1 and position <= 0:  # Buy signal
                    if capital > 10:  # Minimum trade size
                        max_investment = capital * 0.95
                        shares_to_buy = int(max_investment // current_price)
                        
                        if shares_to_buy > 0:
                            execution_price = current_price * (1 + self.slippage)
                            trade_cost = shares_to_buy * execution_price
                            commission_cost = trade_cost * self.commission
                            total_cost = trade_cost + commission_cost
                            
                            if total_cost <= capital:
                                capital -= total_cost
                                position = shares_to_buy
                                
                                trades.append({
                                    'date': current_date,
                                    'action': 'BUY',
                                    'shares': shares_to_buy,
                                    'price': execution_price,
                                    'cost': total_cost
                                })
                                
                                if len(trades) <= 5:  # Debug first few trades
                                    print(f"🔴 BUY: {shares_to_buy} shares at ${execution_price:.2f}")
                
                elif signal == -1 and position > 0:  # Sell signal
                    execution_price = current_price * (1 - self.slippage)
                    trade_revenue = position * execution_price
                    commission_cost = trade_revenue * self.commission
                    net_revenue = trade_revenue - commission_cost
                    
                    capital += net_revenue
                    
                    trades.append({
                        'date': current_date,
                        'action': 'SELL',
                        'shares': position,
                        'price': execution_price,
                        'revenue': net_revenue
                    })
                    
                    if len(trades) <= 5:  # Debug first few trades
                        print(f"🔵 SELL: {position} shares at ${execution_price:.2f}")
                    
                    position = 0
                    
            except Exception as e:
                print(f"⚠️ Warning: Error processing signal {i}: {e}")
                continue
        
        # Calculate final metrics with robust price handling
        final_price = float(prices[-1]) if len(prices) > 0 else 100.0
        final_portfolio_value = capital + (position * final_price)
        total_return = (final_portfolio_value - self.initial_capital) / self.initial_capital
        
        # Additional metrics
        if len(daily_returns) > 0:
            annual_return = np.mean(daily_returns) * 252
            annual_volatility = np.std(daily_returns) * np.sqrt(252)
            sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0
            
            # Maximum drawdown
            if len(portfolio_values) > 0:
                peak = np.maximum.accumulate(portfolio_values)
                drawdown = (np.array(portfolio_values) - peak) / peak
                max_drawdown = np.min(drawdown)
            else:
                max_drawdown = 0
        else:
            annual_return = 0
            annual_volatility = 0
            sharpe_ratio = 0
            max_drawdown = 0
        
        # Buy-and-hold benchmark
        first_price = float(prices[0]) if len(prices) > 0 else 100.0
        last_price = float(prices[-1]) if len(prices) > 0 else 100.0
        buy_hold_return = (last_price - first_price) / first_price
        
        results = {
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_trades': len(trades),
            'final_portfolio_value': final_portfolio_value,
            'buy_hold_return': buy_hold_return,
            'excess_return': total_return - buy_hold_return,
            'portfolio_values': portfolio_values,
            'trades': trades,
            'daily_returns': daily_returns
        }
        
        # Debug output for fold
        print(f"📊 Fold {fold_num} Results:")
        print(f"   Initial: ${self.initial_capital:.2f} → Final: ${final_portfolio_value:.2f}")
        print(f"   Return: {total_return:.2%} | Trades: {len(trades)} | B&H: {buy_hold_return:.2%}")
        
        return results
    
    def _aggregate_results(self, results):
        """Aggregate results from walk-forward validation"""
        total_returns = [r['total_return'] for r in results]
        annual_returns = [r['annual_return'] for r in results]
        sharpe_ratios = [r['sharpe_ratio'] for r in results]
        max_drawdowns = [r['max_drawdown'] for r in results]
        excess_returns = [r['excess_return'] for r in results]
        
        aggregated = {
            'num_folds': len(results),
            'avg_total_return': np.mean(total_returns),
            'std_total_return': np.std(total_returns),
            'avg_annual_return': np.mean(annual_returns),
            'avg_sharpe_ratio': np.mean(sharpe_ratios),
            'avg_max_drawdown': np.mean(max_drawdowns),
            'avg_excess_return': np.mean(excess_returns),
            'positive_periods': sum(1 for r in total_returns if r > 0),
            'consistency_ratio': sum(1 for r in total_returns if r > 0) / len(total_returns),
            'best_period': max(total_returns),
            'worst_period': min(total_returns)
        }
        
        # Statistical significance test
        t_stat, p_value = stats.ttest_1samp(excess_returns, 0)
        aggregated['statistical_significance'] = {
            't_statistic': t_stat,
            'p_value': p_value,
            'is_significant': p_value < 0.05
        }
        
        return aggregated
    
    def run_backtest(self, file_path, output_dir):
        """Run comprehensive backtesting"""
        print("=" * 80)
        print("🎯 FOCUSED LSTM STRATEGY BACKTESTING")
        print("=" * 80)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Load and prepare data
        df = self.load_and_prepare_data(file_path)
        
        # Run walk-forward validation
        fold_results, aggregated_results = self.walk_forward_validation(df)
        
        # Generate report
        self._generate_report(fold_results, aggregated_results, output_dir, timestamp)
        
        # Save results
        self._save_results(fold_results, aggregated_results, output_dir, timestamp)
        
        return fold_results, aggregated_results
    
    def _generate_report(self, fold_results, aggregated_results, output_dir, timestamp):
        """Generate comprehensive backtest report"""
        report_file = os.path.join(output_dir, f'focused_lstm_backtest_{timestamp}.txt')
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("FOCUSED LSTM STRATEGY BACKTESTING REPORT\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Number of Folds: {aggregated_results['num_folds']}\n\n")
            
            f.write("STRATEGY CONFIGURATION:\n")
            f.write("-" * 30 + "\n")
            f.write("📈 Focused Indicators: RSI, ROC, Stochastic, TSI, OBV, MFI, PVT, TEMA, MACD, KAMA, ATR, BB, Ulcer\n")
            f.write("🤖 Model: LSTM [100, 50] units, 0.2 dropout\n")
            f.write("📊 Sequence Length: 60 days\n")
            f.write("🎯 Target Horizon: 5 days\n")
            f.write(f"💰 Initial Capital: ${self.initial_capital:.2f}\n")
            f.write(f"📊 Commission: {self.commission:.3%}\n")
            f.write(f"⚡ Slippage: {self.slippage:.3%}\n\n")
            
            f.write("PERFORMANCE METRICS:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Average Total Return: {aggregated_results['avg_total_return']:.2%}\n")
            f.write(f"Standard Deviation: {aggregated_results['std_total_return']:.2%}\n")
            f.write(f"Average Annual Return: {aggregated_results['avg_annual_return']:.2%}\n")
            f.write(f"Average Sharpe Ratio: {aggregated_results['avg_sharpe_ratio']:.3f}\n")
            f.write(f"Average Max Drawdown: {aggregated_results['avg_max_drawdown']:.2%}\n")
            f.write(f"Average Excess Return: {aggregated_results['avg_excess_return']:.2%}\n\n")
            
            f.write("CONSISTENCY METRICS:\n")
            f.write("-" * 25 + "\n")
            f.write(f"Positive Periods: {aggregated_results['positive_periods']}/{aggregated_results['num_folds']}\n")
            f.write(f"Consistency Ratio: {aggregated_results['consistency_ratio']:.2%}\n")
            f.write(f"Best Period: {aggregated_results['best_period']:.2%}\n")
            f.write(f"Worst Period: {aggregated_results['worst_period']:.2%}\n\n")
            
            # Statistical significance
            sig = aggregated_results['statistical_significance']
            f.write("STATISTICAL SIGNIFICANCE:\n")
            f.write("-" * 30 + "\n")
            f.write(f"T-Statistic: {sig['t_statistic']:.3f}\n")
            f.write(f"P-Value: {sig['p_value']:.4f}\n")
            f.write(f"Significant (p < 0.05): {'Yes' if sig['is_significant'] else 'No'}\n\n")
        
        print(f"📄 Backtest report saved: {report_file}")
    
    def _save_results(self, fold_results, aggregated_results, output_dir, timestamp):
        """Save detailed results to CSV files"""
        # Save fold-by-fold results
        fold_summary = []
        for i, result in enumerate(fold_results, 1):
            fold_summary.append({
                'fold': i,
                'test_start': result['test_start'],
                'test_end': result['test_end'],
                'total_return': result['total_return'],
                'annual_return': result['annual_return'],
                'sharpe_ratio': result['sharpe_ratio'],
                'max_drawdown': result['max_drawdown'],
                'total_trades': result['total_trades'],
                'excess_return': result['excess_return']
            })
        
        fold_df = pd.DataFrame(fold_summary)
        fold_file = os.path.join(output_dir, f'focused_lstm_folds_{timestamp}.csv')
        fold_df.to_csv(fold_file, index=False)
        
        # Save aggregated results
        agg_df = pd.DataFrame([aggregated_results])
        agg_file = os.path.join(output_dir, f'focused_lstm_summary_{timestamp}.csv')
        agg_df.to_csv(agg_file, index=False)
        
        print(f"📊 Results saved:")
        print(f"   Fold results: {fold_file}")
        print(f"   Summary: {agg_file}")

def main():
    """Main function to run focused LSTM backtesting"""
    # Setup paths
    base_dir = os.path.dirname(os.path.dirname(__file__))
    data_file = os.path.join(base_dir, 'data', 'raw', 'priceData5Year.csv')
    output_dir = os.path.join(base_dir, 'output', 'backtest_focused')
    
    # Initialize backtester
    backtester = FocusedLSTMBacktester(
        initial_capital=100,
        commission=0.001,
        slippage=0.0005
    )
    
    try:
        # Run backtesting
        fold_results, aggregated_results = backtester.run_backtest(data_file, output_dir)
        
        # Print summary
        print("\n" + "=" * 60)
        print("🎯 FOCUSED LSTM BACKTESTING SUMMARY")
        print("=" * 60)
        print(f"✅ Completed {aggregated_results['num_folds']} folds")
        print(f"📈 Average Return: {aggregated_results['avg_total_return']:.2%}")
        print(f"📊 Average Sharpe: {aggregated_results['avg_sharpe_ratio']:.3f}")
        print(f"🎯 Consistency: {aggregated_results['consistency_ratio']:.2%}")
        print(f"📉 Avg Max Drawdown: {aggregated_results['avg_max_drawdown']:.2%}")
        
        significance = "✅ Significant" if aggregated_results['statistical_significance']['is_significant'] else "❌ Not Significant"
        print(f"📊 Statistical Significance: {significance}")
        
        print(f"\n📁 All results saved to: {output_dir}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Backtesting failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
