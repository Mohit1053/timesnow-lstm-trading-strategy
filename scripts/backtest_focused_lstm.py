#!/usr/bin/env python3
"""
Comprehensive Backtesting Script for Focused LSTM Trading Strategy

This script provides detailed backtesting capabilities including:
- Walk-forward validation
- Multiple timeframe analysis
- Risk-adjusted performance metrics
- Detailed portfolio simulation
- Statistical significance testing
- Comparative analysis with buy-and-hold
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
from sklearn.metrics import precision_score, recall_score, f1_score

class LSTMBacktester:
    """Comprehensive backtesting framework for LSTM trading strategies"""
    
    def __init__(self, initial_capital=100000, commission=0.001, slippage=0.0005):
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
        self.results = {}
        
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
            
        else:
            # Original format validation
            required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
        
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
                # Train model
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
                strategy.build_model(lstm_units=[100, 50], dropout_rate=0.2)
                strategy.train_model(epochs=30, batch_size=32, validation_split=0.2)
                
                # Make predictions to set y_pred and y_actual
                strategy.make_predictions()
                
                # Generate signals for test period using full dataframe context
                # but limit signals to test period indices
                full_data_copy = df.copy()
                signals_df = strategy.generate_trading_signals(full_data_copy)
                
                # Filter signals to only include test period
                test_start_date = df.index[test_start]
                test_end_date = df.index[test_end-1]
                
                if isinstance(signals_df, list) and len(signals_df) > 0:
                    # Convert list to DataFrame if needed
                    if isinstance(signals_df[0], dict):
                        signals_df = pd.DataFrame(signals_df)
                        if 'date' in signals_df.columns:
                            signals_df['date'] = pd.to_datetime(signals_df['date'])
                            signals_df = signals_df[
                                (signals_df['date'] >= test_start_date) & 
                                (signals_df['date'] <= test_end_date)
                            ]
                
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
        aggregated_results = self._aggregate_walk_forward_results(results)
        
        print(f"\n✅ Walk-forward validation completed: {len(results)} successful folds")
        return results, aggregated_results
    
    def _backtest_signals(self, price_data, signals_df, fold_num=None):
        """
        Backtest trading signals with detailed portfolio simulation
        
        Args:
            price_data (pd.DataFrame): Original price data
            signals_df (pd.DataFrame): Generated trading signals
            fold_num (int): Fold number for walk-forward validation
        """
        print(f"🔍 Debug: price_data columns: {price_data.columns.tolist()}")
        print(f"🔍 Debug: signals_df columns: {signals_df.columns.tolist()}")
        print(f"🔍 Debug: signals_df shape: {signals_df.shape}")
        print(f"🔍 Debug: price_data shape: {price_data.shape}")
        
        # Initialize portfolio tracking
        capital = self.initial_capital
        position = 0
        trades = []
        portfolio_values = []
        daily_returns = []
        
        # Track portfolio for each day
        # Handle different signal date formats
        if 'Date' in signals_df.columns:
            signal_dates_raw = signals_df['Date']
            # Check if dates are synthetic (Day_0, Day_1, etc.)
            if signal_dates_raw.iloc[0].startswith('Day_'):
                # Map synthetic dates to actual price data dates
                # Use the last len(signals_df) dates from price_data
                available_dates = price_data.index[-len(signals_df):]
                signal_dates = available_dates
            else:
                # Try to parse actual dates
                signal_dates = pd.to_datetime(signal_dates_raw)
        elif 'Start_Date' in signals_df.columns:
            signal_dates = pd.to_datetime(signals_df['Start_Date'])
        else:
            # No date columns, map to end of price data chronologically
            available_dates = price_data.index[-len(signals_df):]
            signal_dates = available_dates
        
        # Iterate through signals with proper date mapping
        for i, (signal_idx, row) in enumerate(signals_df.iterrows()):
            try:
                # Get the corresponding date with safe conversion
                if i < len(signal_dates):
                    if hasattr(signal_dates, 'iloc'):
                        date = signal_dates.iloc[i]
                    else:
                        date = signal_dates[i]
                    
                    # Ensure date is a proper datetime
                    if isinstance(date, str):
                        date = pd.to_datetime(date)
                    elif hasattr(date, 'to_pydatetime'):
                        date = date.to_pydatetime()
                else:
                    continue
                
                # Get price from the original price data with safe conversion
                price = None
                if date in price_data.index:
                    if 'Close' in price_data.columns:
                        price_value = price_data.loc[date, 'Close']
                        price = float(price_value) if hasattr(price_value, '__float__') or isinstance(price_value, (int, float)) else float(price_value.iloc[0]) if hasattr(price_value, 'iloc') else 100.0
                    else:
                        # Try alternative column names
                        close_col = None
                        for col in ['Close', 'close', 'CLOSE', 'AdjustedClosePrice']:
                            if col in price_data.columns:
                                close_col = col
                                break
                        if close_col:
                            price_value = price_data.loc[date, close_col]
                            price = float(price_value) if hasattr(price_value, '__float__') or isinstance(price_value, (int, float)) else float(price_value.iloc[0]) if hasattr(price_value, 'iloc') else 100.0
                        else:
                            raise ValueError(f"No close price column found in price_data. Available columns: {price_data.columns.tolist()}")
                else:
                    # Find closest date in price data
                    closest_date = price_data.index[price_data.index.get_indexer([date], method='nearest')[0]]
                    if 'Close' in price_data.columns:
                        price_value = price_data.loc[closest_date, 'Close']
                        price = float(price_value) if hasattr(price_value, '__float__') or isinstance(price_value, (int, float)) else float(price_value.iloc[0]) if hasattr(price_value, 'iloc') else 100.0
                    else:
                        # Try alternative column names
                        close_col = None
                        for col in ['Close', 'close', 'CLOSE', 'AdjustedClosePrice']:
                            if col in price_data.columns:
                                close_col = col
                                break
                        if close_col:
                            price_value = price_data.loc[closest_date, close_col]
                            price = float(price_value) if hasattr(price_value, '__float__') or isinstance(price_value, (int, float)) else float(price_value.iloc[0]) if hasattr(price_value, 'iloc') else 100.0
                        else:
                            raise ValueError(f"No close price column found in price_data. Available columns: {price_data.columns.tolist()}")
                
                # Ensure we have a valid price
                if price is None:
                    price = 100.0
                    
            except Exception as e:
                print(f"⚠️ Warning: Could not get price for signal {i}: {e}")
                # Use previous price if available, otherwise skip
                if i > 0 and len(portfolio_values) > 0:
                    price = portfolio_values[-1] / max(1, position) if position != 0 else 100  # Default price
                else:
                    price = 100  # Default starting price
                    
            signal = row['Signal'] if 'Signal' in row else (1 if row.get('Signal_Type') == 'BUY' else -1 if row.get('Signal_Type') == 'SELL' else 0)
            
            # Calculate current portfolio value
            portfolio_value = capital + (position * price)
            portfolio_values.append(portfolio_value)
            
            # Calculate daily return
            if i > 0:
                daily_return = (portfolio_value - portfolio_values[i-1]) / portfolio_values[i-1]
                # Ensure daily_return is a scalar
                if isinstance(daily_return, (list, tuple, np.ndarray)):
                    daily_return = float(daily_return[0]) if len(daily_return) > 0 else 0.0
                else:
                    daily_return = float(daily_return)
                daily_returns.append(daily_return)
            
            # Execute trades based on signals
            if signal == 1 and position <= 0:  # Buy signal
                if capital > 1000:  # Minimum trade size
                    # Calculate shares to buy (use 95% of available capital)
                    max_investment = capital * 0.95
                    shares_to_buy = int(max_investment // price)
                    
                    if shares_to_buy > 0:
                        # Apply slippage and commission
                        execution_price = price * (1 + self.slippage)
                        trade_cost = shares_to_buy * execution_price
                        commission_cost = trade_cost * self.commission
                        total_cost = trade_cost + commission_cost
                        
                        if total_cost <= capital:
                            capital -= total_cost
                            position = shares_to_buy
                            
                            trades.append({
                                'date': date,
                                'action': 'BUY',
                                'shares': shares_to_buy,
                                'price': execution_price,
                                'cost': total_cost,
                                'commission': commission_cost
                            })
            
            elif signal == -1 and position > 0:  # Sell signal
                # Apply slippage and commission
                execution_price = price * (1 - self.slippage)
                trade_revenue = position * execution_price
                commission_cost = trade_revenue * self.commission
                net_revenue = trade_revenue - commission_cost
                
                capital += net_revenue
                
                trades.append({
                    'date': date,
                    'action': 'SELL',
                    'shares': position,
                    'price': execution_price,
                    'revenue': net_revenue,
                    'commission': commission_cost
                })
                
                position = 0
        
        # Final portfolio value - get final price from price_data
        if 'Close' in price_data.columns:
            final_price = price_data['Close'].iloc[-1]
        else:
            # Try alternative column names
            close_col = None
            for col in ['Close', 'close', 'CLOSE', 'AdjustedClosePrice']:
                if col in price_data.columns:
                    close_col = col
                    break
            if close_col:
                final_price = price_data[close_col].iloc[-1]
            else:
                # Use the last calculated price from the loop
                final_price = price if 'price' in locals() else 100
        final_portfolio_value = capital + (position * final_price)
        
        # Calculate performance metrics
        total_return = (final_portfolio_value - self.initial_capital) / self.initial_capital
        
        # Calculate additional metrics
        if len(daily_returns) > 0:
            # Ensure all daily returns are scalar values
            daily_returns_clean = []
            for dr in daily_returns:
                if isinstance(dr, (list, tuple, np.ndarray)):
                    dr_clean = float(dr[0]) if len(dr) > 0 else 0.0
                else:
                    dr_clean = float(dr)
                daily_returns_clean.append(dr_clean)
            
            daily_returns = np.array(daily_returns_clean)
            annual_return = np.mean(daily_returns) * 252
            annual_volatility = np.std(daily_returns) * np.sqrt(252)
            sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0
            
            # Calculate maximum drawdown
            peak = np.maximum.accumulate(portfolio_values)
            drawdown = (np.array(portfolio_values) - peak) / peak
            max_drawdown = np.min(drawdown)
            
            # Calculate win rate
            if len(trades) > 1:
                winning_trades = 0
                for i in range(1, len(trades), 2):  # Every second trade should be a sell
                    if i < len(trades) and trades[i]['action'] == 'SELL':
                        if i > 0 and trades[i-1]['action'] == 'BUY':
                            profit = trades[i]['revenue'] - trades[i-1]['cost']
                            if profit > 0:
                                winning_trades += 1
                
                win_rate = winning_trades / (len(trades) // 2) if len(trades) > 1 else 0
            else:
                win_rate = 0
        else:
            annual_return = 0
            annual_volatility = 0
            sharpe_ratio = 0
            max_drawdown = 0
            win_rate = 0
        
        # Calculate buy-and-hold benchmark using price_data
        if 'Close' in price_data.columns:
            first_price = price_data['Close'].iloc[0]
            last_price = price_data['Close'].iloc[-1]
        else:
            # Try alternative column names
            close_col = None
            for col in ['Close', 'close', 'CLOSE', 'AdjustedClosePrice']:
                if col in price_data.columns:
                    close_col = col
                    break
            if close_col:
                first_price = price_data[close_col].iloc[0]
                last_price = price_data[close_col].iloc[-1]
            else:
                # Default to no return if no price column found
                first_price = last_price = 100
        
        buy_hold_return = (last_price - first_price) / first_price
        
        # Compile results
        results = {
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': len(trades),
            'final_portfolio_value': final_portfolio_value,
            'buy_hold_return': buy_hold_return,
            'excess_return': total_return - buy_hold_return,
            'portfolio_values': portfolio_values,
            'trades': trades,
            'daily_returns': daily_returns
        }
        
        return results
    
    def _aggregate_walk_forward_results(self, results):
        """Aggregate results from walk-forward validation"""
        if not results:
            return {}
        
        # Calculate aggregate metrics
        total_returns = [r['total_return'] for r in results]
        annual_returns = [r['annual_return'] for r in results]
        sharpe_ratios = [r['sharpe_ratio'] for r in results]
        max_drawdowns = [r['max_drawdown'] for r in results]
        win_rates = [r['win_rate'] for r in results]
        excess_returns = [r['excess_return'] for r in results]
        
        aggregated = {
            'num_folds': len(results),
            'avg_total_return': np.mean(total_returns),
            'std_total_return': np.std(total_returns),
            'avg_annual_return': np.mean(annual_returns),
            'avg_sharpe_ratio': np.mean(sharpe_ratios),
            'avg_max_drawdown': np.mean(max_drawdowns),
            'avg_win_rate': np.mean(win_rates),
            'avg_excess_return': np.mean(excess_returns),
            'positive_periods': sum(1 for r in total_returns if r > 0),
            'negative_periods': sum(1 for r in total_returns if r <= 0),
            'best_period': max(total_returns),
            'worst_period': min(total_returns),
            'consistency_ratio': sum(1 for r in total_returns if r > 0) / len(total_returns)
        }
        
        # Statistical significance test
        t_stat, p_value = stats.ttest_1samp(excess_returns, 0)
        aggregated['statistical_significance'] = {
            't_statistic': t_stat,
            'p_value': p_value,
            'is_significant': p_value < 0.05
        }
        
        return aggregated
    
    def comprehensive_backtest(self, file_path, output_dir):
        """
        Run comprehensive backtesting including multiple analyses
        
        Args:
            file_path (str): Path to price data file
            output_dir (str): Directory to save results
        """
        print("=" * 80)
        print("🚀 COMPREHENSIVE LSTM BACKTESTING")
        print("=" * 80)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Load and prepare data
        df = self.load_and_prepare_data(file_path)
        
        # Run walk-forward validation
        fold_results, aggregated_results = self.walk_forward_validation(df)
        
        # Generate detailed reports
        self._generate_backtest_report(fold_results, aggregated_results, output_dir, timestamp)
        
        # Create visualizations
        self._create_backtest_visualizations(fold_results, aggregated_results, output_dir, timestamp)
        
        # Save detailed results
        self._save_detailed_results(fold_results, aggregated_results, output_dir, timestamp)
        
        return fold_results, aggregated_results
    
    def _generate_backtest_report(self, fold_results, aggregated_results, output_dir, timestamp):
        """Generate comprehensive backtest report"""
        report_file = os.path.join(output_dir, f'backtest_report_{timestamp}.txt')
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("COMPREHENSIVE LSTM BACKTESTING REPORT\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Number of Folds: {aggregated_results['num_folds']}\n\n")
            
            f.write("STRATEGY CONFIGURATION:\n")
            f.write("-" * 30 + "\n")
            f.write("📈 Indicators: RSI, ROC, Stochastic, TSI, OBV, MFI, PVT, TEMA, MACD, KAMA, ATR, BB, Ulcer\n")
            f.write("🤖 Model: LSTM [100, 50] units, 0.2 dropout\n")
            f.write("📊 Sequence Length: 60 days\n")
            f.write("🎯 Target Horizon: 5 days\n")
            f.write(f"💰 Initial Capital: ${self.initial_capital:,.2f}\n")
            f.write(f"📊 Commission: {self.commission:.3%}\n")
            f.write(f"⚡ Slippage: {self.slippage:.3%}\n\n")
            
            f.write("AGGREGATED PERFORMANCE METRICS:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Average Total Return: {aggregated_results['avg_total_return']:.2%}\n")
            f.write(f"Standard Deviation: {aggregated_results['std_total_return']:.2%}\n")
            f.write(f"Average Annual Return: {aggregated_results['avg_annual_return']:.2%}\n")
            f.write(f"Average Sharpe Ratio: {aggregated_results['avg_sharpe_ratio']:.3f}\n")
            f.write(f"Average Max Drawdown: {aggregated_results['avg_max_drawdown']:.2%}\n")
            f.write(f"Average Win Rate: {aggregated_results['avg_win_rate']:.2%}\n")
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
            
            f.write("INDIVIDUAL FOLD RESULTS:\n")
            f.write("-" * 30 + "\n")
            f.write("Fold | Period                    | Return  | Sharpe | Max DD | Win Rate\n")
            f.write("-" * 70 + "\n")
            
            for i, result in enumerate(fold_results, 1):
                start_date = result['test_start'].strftime('%Y-%m-%d')
                end_date = result['test_end'].strftime('%Y-%m-%d')
                f.write(f"{i:4d} | {start_date} to {end_date} | "
                       f"{result['total_return']:6.2%} | {result['sharpe_ratio']:6.3f} | "
                       f"{result['max_drawdown']:6.2%} | {result['win_rate']:7.2%}\n")
        
        print(f"📄 Backtest report saved: {report_file}")
    
    def _create_backtest_visualizations(self, fold_results, aggregated_results, output_dir, timestamp):
        """Create comprehensive visualizations"""
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create a comprehensive figure
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Returns Distribution
        plt.subplot(3, 3, 1)
        returns = [r['total_return'] for r in fold_results]
        plt.hist(returns, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(np.mean(returns), color='red', linestyle='--', label=f'Mean: {np.mean(returns):.2%}')
        plt.xlabel('Total Return')
        plt.ylabel('Frequency')
        plt.title('Distribution of Returns Across Folds')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. Rolling Sharpe Ratios
        plt.subplot(3, 3, 2)
        sharpe_ratios = [r['sharpe_ratio'] for r in fold_results]
        fold_numbers = list(range(1, len(fold_results) + 1))
        plt.plot(fold_numbers, sharpe_ratios, marker='o', linewidth=2, markersize=4)
        plt.axhline(np.mean(sharpe_ratios), color='red', linestyle='--', 
                   label=f'Average: {np.mean(sharpe_ratios):.3f}')
        plt.xlabel('Fold Number')
        plt.ylabel('Sharpe Ratio')
        plt.title('Sharpe Ratio Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 3. Drawdown Analysis
        plt.subplot(3, 3, 3)
        max_drawdowns = [r['max_drawdown'] for r in fold_results]
        plt.plot(fold_numbers, max_drawdowns, marker='s', color='red', linewidth=2, markersize=4)
        plt.axhline(np.mean(max_drawdowns), color='darkred', linestyle='--',
                   label=f'Average: {np.mean(max_drawdowns):.2%}')
        plt.xlabel('Fold Number')
        plt.ylabel('Maximum Drawdown')
        plt.title('Maximum Drawdown Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 4. Win Rate Analysis
        plt.subplot(3, 3, 4)
        win_rates = [r['win_rate'] for r in fold_results]
        plt.bar(fold_numbers, win_rates, alpha=0.7, color='green')
        plt.axhline(np.mean(win_rates), color='darkgreen', linestyle='--',
                   label=f'Average: {np.mean(win_rates):.2%}')
        plt.xlabel('Fold Number')
        plt.ylabel('Win Rate')
        plt.title('Win Rate by Fold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 5. Strategy vs Buy-and-Hold
        plt.subplot(3, 3, 5)
        strategy_returns = [r['total_return'] for r in fold_results]
        buy_hold_returns = [r['buy_hold_return'] for r in fold_results]
        
        plt.scatter(buy_hold_returns, strategy_returns, alpha=0.7, s=50)
        # Add diagonal line
        min_val = min(min(strategy_returns), min(buy_hold_returns))
        max_val = max(max(strategy_returns), max(buy_hold_returns))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7, label='Equal Performance')
        
        plt.xlabel('Buy-and-Hold Return')
        plt.ylabel('Strategy Return')
        plt.title('Strategy vs Buy-and-Hold Returns')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 6. Excess Returns
        plt.subplot(3, 3, 6)
        excess_returns = [r['excess_return'] for r in fold_results]
        colors = ['green' if x > 0 else 'red' for x in excess_returns]
        plt.bar(fold_numbers, excess_returns, color=colors, alpha=0.7)
        plt.axhline(0, color='black', linestyle='-', alpha=0.5)
        plt.axhline(np.mean(excess_returns), color='blue', linestyle='--',
                   label=f'Average: {np.mean(excess_returns):.2%}')
        plt.xlabel('Fold Number')
        plt.ylabel('Excess Return')
        plt.title('Excess Return Over Buy-and-Hold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 7. Cumulative Performance
        plt.subplot(3, 3, 7)
        cumulative_returns = np.cumprod(1 + np.array(strategy_returns))
        cumulative_buy_hold = np.cumprod(1 + np.array(buy_hold_returns))
        
        plt.plot(fold_numbers, cumulative_returns, label='Strategy', linewidth=2)
        plt.plot(fold_numbers, cumulative_buy_hold, label='Buy-and-Hold', linewidth=2)
        plt.xlabel('Fold Number')
        plt.ylabel('Cumulative Return')
        plt.title('Cumulative Performance Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 8. Return vs Risk
        plt.subplot(3, 3, 8)
        annual_returns = [r['annual_return'] for r in fold_results]
        annual_volatilities = [r['annual_volatility'] for r in fold_results]
        
        plt.scatter(annual_volatilities, annual_returns, alpha=0.7, s=50, c=sharpe_ratios, 
                   cmap='viridis')
        plt.colorbar(label='Sharpe Ratio')
        plt.xlabel('Annual Volatility')
        plt.ylabel('Annual Return')
        plt.title('Risk-Return Profile')
        plt.grid(True, alpha=0.3)
        
        # 9. Performance Metrics Summary
        plt.subplot(3, 3, 9)
        metrics = ['Avg Return', 'Avg Sharpe', 'Avg Win Rate', 'Consistency']
        values = [
            aggregated_results['avg_total_return'] * 100,
            aggregated_results['avg_sharpe_ratio'],
            aggregated_results['avg_win_rate'] * 100,
            aggregated_results['consistency_ratio'] * 100
        ]
        
        bars = plt.bar(metrics, values, color=['blue', 'green', 'orange', 'purple'], alpha=0.7)
        plt.ylabel('Value')
        plt.title('Key Performance Metrics')
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.1f}%' if metrics[bars.index(bar)] != 'Avg Sharpe' else f'{value:.3f}',
                    ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save the plot
        plot_file = os.path.join(output_dir, f'backtest_analysis_{timestamp}.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 Backtest visualizations saved: {plot_file}")
    
    def _save_detailed_results(self, fold_results, aggregated_results, output_dir, timestamp):
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
                'annual_volatility': result['annual_volatility'],
                'sharpe_ratio': result['sharpe_ratio'],
                'max_drawdown': result['max_drawdown'],
                'win_rate': result['win_rate'],
                'total_trades': result['total_trades'],
                'buy_hold_return': result['buy_hold_return'],
                'excess_return': result['excess_return']
            })
        
        fold_df = pd.DataFrame(fold_summary)
        fold_file = os.path.join(output_dir, f'fold_results_{timestamp}.csv')
        fold_df.to_csv(fold_file, index=False)
        
        # Save aggregated results
        agg_df = pd.DataFrame([aggregated_results])
        agg_file = os.path.join(output_dir, f'aggregated_results_{timestamp}.csv')
        agg_df.to_csv(agg_file, index=False)
        
        print(f"📊 Detailed results saved:")
        print(f"   Fold results: {fold_file}")
        print(f"   Aggregated results: {agg_file}")

def main():
    """Main function to run comprehensive backtesting"""
    # Setup paths
    base_dir = os.path.dirname(os.path.dirname(__file__))
    data_file = os.path.join(base_dir, 'data', 'raw', 'priceData5Year.csv')
    output_dir = os.path.join(base_dir, 'output', 'backtest')
    
    # Initialize backtester
    backtester = LSTMBacktester(
        initial_capital=100,
        commission=0.001,  # 0.1%
        slippage=0.0005    # 0.05%
    )
    
    try:
        # Run comprehensive backtesting
        fold_results, aggregated_results = backtester.comprehensive_backtest(data_file, output_dir)
        
        # Print summary
        print("\n" + "=" * 60)
        print("🎯 BACKTESTING SUMMARY")
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
