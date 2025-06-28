"""
Technical Indicators Generator for Stock Data
This script creates a comprehensive dataset with various technical indicators
from stock price data.
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def calculate_sma(data, window):
    """Simple Moving Average"""
    return data.rolling(window=window).mean()

def calculate_ema(data, window):
    """Exponential Moving Average"""
    return data.ewm(span=window).mean()

def calculate_macd(data, fast=12, slow=26, signal=9):
    """Moving Average Convergence Divergence"""
    ema_fast = calculate_ema(data, fast)
    ema_slow = calculate_ema(data, slow)
    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def calculate_rsi(data, window=14):
    """Relative Strength Index with Wilder's smoothing"""
    delta = data.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # Use Wilder's smoothing (exponential smoothing with alpha = 1/window)
    alpha = 1.0 / window
    avg_gain = gain.ewm(alpha=alpha, adjust=False).mean()
    avg_loss = loss.ewm(alpha=alpha, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_bollinger_bands(data, window=20, std_dev=2):
    """Bollinger Bands"""
    sma = calculate_sma(data, window)
    std = data.rolling(window=window).std()
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    return upper_band, sma, lower_band

def calculate_stochastic(high, low, close, k_window=14, d_window=3):
    """Stochastic Oscillator"""
    lowest_low = low.rolling(window=k_window).min()
    highest_high = high.rolling(window=k_window).max()
    k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    d_percent = k_percent.rolling(window=d_window).mean()
    return k_percent, d_percent

def calculate_atr(high, low, close, window=14):
    """Average True Range"""
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=window).mean()
    return atr

def calculate_adx(high, low, close, window=14):
    """Average Directional Index with Wilder's smoothing"""
    # Calculate True Range
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Calculate Directional Movement using pandas operations to maintain index alignment
    high_diff = high - high.shift()
    low_diff = low.shift() - low
    
    dm_plus = pd.Series(np.where(high_diff > low_diff, np.maximum(high_diff, 0), 0), index=high.index)
    dm_minus = pd.Series(np.where(low_diff > high_diff, np.maximum(low_diff, 0), 0), index=high.index)
    
    # Use Wilder's smoothing (exponential smoothing with alpha = 1/window)
    alpha = 1.0 / window
    tr_smooth = tr.ewm(alpha=alpha, adjust=False).mean()
    dm_plus_smooth = dm_plus.ewm(alpha=alpha, adjust=False).mean()
    dm_minus_smooth = dm_minus.ewm(alpha=alpha, adjust=False).mean()
    
    # Calculate DI+ and DI- with safe division
    di_plus = 100 * (dm_plus_smooth / tr_smooth.replace(0, np.nan))
    di_minus = 100 * (dm_minus_smooth / tr_smooth.replace(0, np.nan))
    
    # Calculate DX and ADX with safe division
    di_sum = di_plus + di_minus
    dx = 100 * abs(di_plus - di_minus) / di_sum.replace(0, np.nan)
    adx = dx.ewm(alpha=alpha, adjust=False).mean()
    
    return adx, di_plus, di_minus

def calculate_parabolic_sar(high, low, af_start=0.02, af_max=0.2):
    """Parabolic SAR"""
    length = len(high)
    psar = np.zeros(length)
    af = af_start
    trend = 1  # 1 for uptrend, -1 for downtrend
    ep = high.iloc[0]  # extreme point
    
    psar[0] = low.iloc[0]
    
    for i in range(1, length):
        if trend == 1:  # Uptrend
            psar[i] = psar[i-1] + af * (ep - psar[i-1])
            if low.iloc[i] < psar[i]:
                trend = -1
                psar[i] = ep
                ep = low.iloc[i]
                af = af_start
            else:
                if high.iloc[i] > ep:
                    ep = high.iloc[i]
                    af = min(af + af_start, af_max)
        else:  # Downtrend
            psar[i] = psar[i-1] - af * (psar[i-1] - ep)
            if high.iloc[i] > psar[i]:
                trend = 1
                psar[i] = ep
                ep = high.iloc[i]
                af = af_start
            else:
                if low.iloc[i] < ep:
                    ep = low.iloc[i]
                    af = min(af + af_start, af_max)
    
    return pd.Series(psar, index=high.index)

def calculate_ichimoku(high, low, close, tenkan=9, kijun=26, senkou_b=52):
    """Ichimoku Cloud"""
    # Tenkan-sen (Conversion Line)
    tenkan_sen = (high.rolling(window=tenkan).max() + low.rolling(window=tenkan).min()) / 2
    
    # Kijun-sen (Base Line)
    kijun_sen = (high.rolling(window=kijun).max() + low.rolling(window=kijun).min()) / 2
    
    # Senkou Span A (Leading Span A)
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun)
    
    # Senkou Span B (Leading Span B)
    senkou_span_b = ((high.rolling(window=senkou_b).max() + 
                      low.rolling(window=senkou_b).min()) / 2).shift(kijun)
    
    # Chikou Span (Lagging Span)
    chikou_span = close.shift(-kijun)
    
    return tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b, chikou_span

def calculate_roc(data, window=12):
    """Rate of Change"""
    return ((data - data.shift(window)) / data.shift(window)) * 100

def calculate_cci(high, low, close, window=20):
    """Commodity Channel Index"""
    tp = (high + low + close) / 3  # Typical Price
    sma_tp = tp.rolling(window=window).mean()
    mad = tp.rolling(window=window).apply(lambda x: np.abs(x - x.mean()).mean())
    cci = (tp - sma_tp) / (0.015 * mad)
    return cci

def calculate_obv(close, volume):
    """On-Balance Volume with missing data handling"""
    if volume.isna().all() or close.isna().all():
        return pd.Series(index=close.index, dtype=float)
    
    # Clean data - forward fill missing values
    clean_close = close.fillna(method='ffill').fillna(method='bfill')
    clean_volume = volume.fillna(0)  # Fill missing volume with 0
    
    obv = np.zeros(len(clean_close))
    if len(clean_volume) > 0:
        obv[0] = clean_volume.iloc[0]
    
    for i in range(1, len(clean_close)):
        if pd.isna(clean_close.iloc[i]) or pd.isna(clean_close.iloc[i-1]):
            obv[i] = obv[i-1]  # No change if price data is missing
        elif clean_close.iloc[i] > clean_close.iloc[i-1]:
            obv[i] = obv[i-1] + clean_volume.iloc[i]
        elif clean_close.iloc[i] < clean_close.iloc[i-1]:
            obv[i] = obv[i-1] - clean_volume.iloc[i]
        else:
            obv[i] = obv[i-1]
    
    return pd.Series(obv, index=close.index)

def calculate_ad_line(high, low, close, volume):
    """Accumulation/Distribution Line with missing data handling"""
    if volume.isna().all() or high.isna().all() or low.isna().all() or close.isna().all():
        return pd.Series(index=close.index, dtype=float)
    
    # Clean data
    clean_high = high.fillna(method='ffill').fillna(method='bfill')
    clean_low = low.fillna(method='ffill').fillna(method='bfill')
    clean_close = close.fillna(method='ffill').fillna(method='bfill')
    clean_volume = volume.fillna(0)
    
    # Calculate Money Flow Multiplier with safe division
    high_low_diff = clean_high - clean_low
    mfm = ((clean_close - clean_low) - (clean_high - clean_close)) / high_low_diff.replace(0, np.nan)
    mfm = mfm.fillna(0)  # Handle division by zero
    
    # Calculate Money Flow Volume
    mfv = mfm * clean_volume
    
    # Calculate A/D Line as cumulative sum
    ad_line = mfv.cumsum()
    return ad_line

def calculate_pivot_points(high, low, close):
    """Pivot Points"""
    pivot = (high + low + close) / 3
    r1 = 2 * pivot - low
    s1 = 2 * pivot - high
    r2 = pivot + (high - low)
    s2 = pivot - (high - low)
    r3 = high + 2 * (pivot - low)
    s3 = low - 2 * (high - pivot)
    
    return pivot, r1, r2, r3, s1, s2, s3

def add_fibonacci_levels(high, low, lookback=50):
    """Fibonacci Retracement Levels"""
    rolling_high = high.rolling(window=lookback).max()
    rolling_low = low.rolling(window=lookback).min()
    diff = rolling_high - rolling_low
    
    fib_236 = rolling_high - 0.236 * diff
    fib_382 = rolling_high - 0.382 * diff
    fib_500 = rolling_high - 0.500 * diff
    fib_618 = rolling_high - 0.618 * diff
    fib_786 = rolling_high - 0.786 * diff
    
    return fib_236, fib_382, fib_500, fib_618, fib_786

def calculate_volume_profile(close, volume, bins=20):
    """Volume Profile - Distribution of volume at different price levels"""
    if len(close) < bins or volume.isna().all() or close.isna().all():
        return pd.Series(index=close.index, dtype=float), pd.Series(index=close.index, dtype=float)
    
    # Clean the data
    valid_mask = ~(close.isna() | volume.isna() | (volume <= 0))
    if valid_mask.sum() < bins:
        return pd.Series(index=close.index, dtype=float), pd.Series(index=close.index, dtype=float)
    
    # Initialize output series
    volume_at_price = pd.Series(index=close.index, dtype=float)
    dominant_price_level = pd.Series(index=close.index, dtype=float)
    
    for i in range(len(close)):
        if i < bins:
            # Not enough data for meaningful volume profile
            volume_at_price.iloc[i] = np.nan
            dominant_price_level.iloc[i] = np.nan
        else:
            # Look at recent period for volume profile
            start_idx = max(0, i - bins)
            recent_close = close.iloc[start_idx:i+1]
            recent_volume = volume.iloc[start_idx:i+1]
            
            # Filter out invalid data
            valid_recent = ~(recent_close.isna() | recent_volume.isna() | (recent_volume <= 0))
            
            if valid_recent.sum() < 5:  # Need at least 5 valid points
                volume_at_price.iloc[i] = np.nan
                dominant_price_level.iloc[i] = np.nan
                continue
            
            valid_close = recent_close[valid_recent]
            valid_volume = recent_volume[valid_recent]
            
            # Create bins for this period
            period_min = valid_close.min()
            period_max = valid_close.max()
            
            if period_max > period_min:
                try:
                    period_bins = np.linspace(period_min, period_max, bins + 1)
                    period_bin_indices = np.digitize(valid_close, period_bins) - 1
                    period_bin_indices = np.clip(period_bin_indices, 0, bins - 1)
                    
                    # Sum volume for each bin
                    bin_volumes = np.zeros(bins)
                    for j, bin_idx in enumerate(period_bin_indices):
                        bin_volumes[bin_idx] += valid_volume.iloc[j]
                    
                    # Find price level with highest volume
                    max_volume_bin = np.argmax(bin_volumes)
                    dominant_price = (period_bins[max_volume_bin] + period_bins[max_volume_bin + 1]) / 2
                    
                    volume_at_price.iloc[i] = bin_volumes[max_volume_bin]
                    dominant_price_level.iloc[i] = dominant_price
                except Exception:
                    volume_at_price.iloc[i] = np.nan
                    dominant_price_level.iloc[i] = np.nan
            else:
                volume_at_price.iloc[i] = valid_volume.sum()
                dominant_price_level.iloc[i] = valid_close.iloc[-1]
    
    return volume_at_price, dominant_price_level

def create_technical_indicators_dataset(file_path, chunk_size=None):
    """
    Main function to create a comprehensive technical indicators dataset
    For very large datasets, set chunk_size to process data in smaller batches
    """
    print("Loading data...")
    
    # Try to read the CSV file with common column names
    try:
        if chunk_size:
            print(f"Processing data in chunks of {chunk_size} rows...")
            # Read first chunk to get column info
            chunk_iter = pd.read_csv(file_path, chunksize=chunk_size)
            first_chunk = next(chunk_iter)
            print(f"First chunk loaded. Shape: {first_chunk.shape}")
            print(f"Columns: {first_chunk.columns.tolist()}")
            
            # Process all chunks
            all_chunks = [first_chunk]
            for i, chunk in enumerate(chunk_iter, 2):
                print(f"Loading chunk {i}...")
                all_chunks.append(chunk)
            
            df = pd.concat(all_chunks, ignore_index=True)
            print(f"All chunks concatenated. Final shape: {df.shape}")
        else:
            df = pd.read_csv(file_path)
            print(f"Data loaded successfully. Shape: {df.shape}")
            print(f"Columns: {df.columns.tolist()}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
    
    # Auto-detect column names (case insensitive)
    columns_map = {}
    for col in df.columns:
        col_lower = col.lower()
        if 'date' in col_lower or 'time' in col_lower:
            columns_map['date'] = col
        elif 'open' in col_lower:
            columns_map['open'] = col
        elif 'high' in col_lower:
            columns_map['high'] = col
        elif 'low' in col_lower:
            columns_map['low'] = col
        elif 'close' in col_lower:
            columns_map['close'] = col
        elif 'volume' in col_lower or 'quantity' in col_lower or 'traded' in col_lower:
            columns_map['volume'] = col
    
    print(f"Detected columns mapping: {columns_map}")
    
    # Ensure we have the required columns
    required_cols = ['open', 'high', 'low', 'close']
    missing_cols = [col for col in required_cols if col not in columns_map]
    if missing_cols:
        print(f"Missing required columns: {missing_cols}")
        return None
    
    # Rename columns for easier access
    df_renamed = df.rename(columns={v: k for k, v in columns_map.items()})
    
    # Convert date column if it exists
    if 'date' in columns_map:
        df_renamed['date'] = pd.to_datetime(df_renamed['date'])
        df_renamed = df_renamed.sort_values('date')
        df_renamed = df_renamed.set_index('date')
    
    # Ensure numeric columns
    numeric_cols = ['open', 'high', 'low', 'close']
    if 'volume' in columns_map:
        numeric_cols.append('volume')
    
    for col in numeric_cols:
        df_renamed[col] = pd.to_numeric(df_renamed[col], errors='coerce')
    
    print("Calculating technical indicators...")
    
    # For multi-company datasets, process by company to reduce memory usage
    if 'companyid' in df_renamed.columns and len(df_renamed['companyid'].unique()) > 1:
        result_df = process_by_company(df_renamed)
    else:
        # Process as single dataset
        result_df = calculate_indicators_for_company(df_renamed)
    
    if result_df is None:
        print("Failed to calculate technical indicators")
        return None
    
    print(f"Technical indicators calculation completed!")
    print(f"Final dataset shape: {result_df.shape}")
    print(f"Number of indicators added: {result_df.shape[1] - df_renamed.shape[1]}")
    
    return result_df

def process_by_company(df):
    """
    Process technical indicators for each company separately to reduce memory usage
    """
    print("Processing data by company groups...")
    
    # Get unique companies
    companies = df['companyid'].unique()
    print(f"Found {len(companies)} companies to process")
    
    processed_dfs = []
    skipped_companies = []
    error_companies = []
    
    for i, company_id in enumerate(companies, 1):
        # print(f"Processing company {i}/{len(companies)}: {company_id}")
        
        # Filter data for this company
        company_data = df[df['companyid'] == company_id].copy()
        
        # Check data quality before processing
        if len(company_data) < 50:
            print(f"  Skipping company {company_id} - insufficient data ({len(company_data)} rows)")
            skipped_companies.append({'company_id': company_id, 'reason': 'insufficient_data', 'rows': len(company_data)})
            continue
        
        # Check for completely missing OHLC data
        required_cols = ['high', 'low', 'close', 'open']
        missing_all = all(company_data[col].isna().all() for col in required_cols)
        if missing_all:
            print(f"  Skipping company {company_id} - all OHLC data is missing")
            skipped_companies.append({'company_id': company_id, 'reason': 'all_ohlc_missing', 'rows': len(company_data)})
            continue
        
        # Sort by date if date column exists
        if 'date' in company_data.columns:
            company_data = company_data.sort_values('date')
        company_data = company_data.reset_index(drop=True)
        
        # Calculate technical indicators for this company
        try:
            processed_company = calculate_indicators_for_company(company_data)
            
            if processed_company is not None:
                processed_dfs.append(processed_company)
                # print(f"  Successfully processed company {company_id}")
            else:
                print(f"  Failed to process company {company_id} - returned None")
                error_companies.append({'company_id': company_id, 'reason': 'processing_failed'})
        except Exception as e:
            print(f"  Error processing company {company_id}: {str(e)}")
            error_companies.append({'company_id': company_id, 'reason': f'exception: {str(e)}'})
        
        # Memory cleanup
        del company_data
        if i % 10 == 0:  # Garbage collect every 10 companies
            import gc
            gc.collect()
    
    # Print summary
    print(f"\nProcessing Summary:")
    print(f"  Successfully processed: {len(processed_dfs)} companies")
    print(f"  Skipped: {len(skipped_companies)} companies")
    print(f"  Errors: {len(error_companies)} companies")
    
    if skipped_companies:
        print(f"\nSkipped companies details:")
        for skip in skipped_companies[:5]:  # Show first 5
            print(f"  - {skip['company_id']}: {skip['reason']} ({skip.get('rows', 'N/A')} rows)")
        if len(skipped_companies) > 5:
            print(f"  ... and {len(skipped_companies) - 5} more")
    
    if error_companies:
        print(f"\nError companies details:")
        for error in error_companies[:5]:  # Show first 5
            print(f"  - {error['company_id']}: {error['reason']}")
        if len(error_companies) > 5:
            print(f"  ... and {len(error_companies) - 5} more")
    
    if processed_dfs:
        print("Combining all company data...")
        final_df = pd.concat(processed_dfs, ignore_index=True)
        return final_df
    else:
        print("No companies were successfully processed!")
        return None

def calculate_indicators_for_company(company_data):
    """
    Calculate technical indicators for a single company's data
    """
    try:
        # Data validation and cleaning
        # print(f"  Data validation for company...")
        
        # Check for missing OHLC data
        required_cols = ['high', 'low', 'close', 'open']
        missing_data_summary = {}
        
        for col in required_cols:
            missing_count = company_data[col].isna().sum()
            missing_pct = (missing_count / len(company_data)) * 100
            missing_data_summary[col] = {'count': missing_count, 'percentage': missing_pct}
            
            if missing_pct > 50:
                print(f"  WARNING: {col} has {missing_pct:.1f}% missing values")
        
        # Forward fill missing values for OHLC data (common practice)
        for col in required_cols:
            if company_data[col].isna().any():
                print(f"  Forward filling missing {col} values...")
                company_data[col] = company_data[col].fillna(method='ffill')
                # If still NaN at the beginning, backward fill
                company_data[col] = company_data[col].fillna(method='bfill')
        
        # Handle volume data separately
        if 'volume' in company_data.columns:
            volume_missing = company_data['volume'].isna().sum()
            if volume_missing > 0:
                print(f"  Volume has {volume_missing} missing values, filling with 0...")
                company_data['volume'] = company_data['volume'].fillna(0)
        
        # Validate data consistency (High >= Low, etc.)
        invalid_data = (company_data['high'] < company_data['low']).sum()
        if invalid_data > 0:
            print(f"  WARNING: Found {invalid_data} rows where High < Low, fixing...")
            # Swap high and low where high < low
            mask = company_data['high'] < company_data['low']
            company_data.loc[mask, ['high', 'low']] = company_data.loc[mask, ['low', 'high']].values
        
        # Check for negative prices
        negative_prices = (company_data[required_cols] < 0).any(axis=1).sum()
        if negative_prices > 0:
            print(f"  WARNING: Found {negative_prices} rows with negative prices, removing...")
            company_data = company_data[(company_data[required_cols] >= 0).all(axis=1)]
        
        # Final check for sufficient data
        if len(company_data) < 30:
            print(f"  Insufficient data after cleaning ({len(company_data)} rows)")
            return None
        
        high = company_data['high']
        low = company_data['low']
        close = company_data['close']
        open_price = company_data['open']
        
        # Create result dataframe
        result_df = company_data.copy()
        
        # Moving Averages
        result_df['SMA_10'] = calculate_sma(close, 10)
        result_df['SMA_20'] = calculate_sma(close, 20)
        result_df['SMA_50'] = calculate_sma(close, 50)
        result_df['EMA_12'] = calculate_ema(close, 12)
        result_df['EMA_26'] = calculate_ema(close, 26)
        result_df['EMA_50'] = calculate_ema(close, 50)
        
        # MACD
        macd_line, signal_line, histogram = calculate_macd(close)
        result_df['MACD'] = macd_line
        result_df['MACD_Signal'] = signal_line
        result_df['MACD_Histogram'] = histogram
        
        # RSI
        result_df['RSI'] = calculate_rsi(close)
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(close)
        result_df['BB_Upper'] = bb_upper
        result_df['BB_Middle'] = bb_middle
        result_df['BB_Lower'] = bb_lower
        result_df['BB_Width'] = bb_upper - bb_lower
        result_df['BB_Position'] = (close - bb_lower) / (bb_upper - bb_lower)
        
        # Stochastic Oscillator
        stoch_k, stoch_d = calculate_stochastic(high, low, close)
        result_df['Stoch_K'] = stoch_k
        result_df['Stoch_D'] = stoch_d
        
        # ATR
        result_df['ATR'] = calculate_atr(high, low, close)
        
        # ADX
        adx, di_plus, di_minus = calculate_adx(high, low, close)
        result_df['ADX'] = adx
        result_df['DI_Plus'] = di_plus
        result_df['DI_Minus'] = di_minus
        
        # Parabolic SAR
        if len(close) > 50:  # Only calculate if sufficient data
            result_df['PSAR'] = calculate_parabolic_sar(high, low)
        else:
            result_df['PSAR'] = np.nan
        
        # Ichimoku Cloud
        tenkan, kijun, senkou_a, senkou_b, chikou = calculate_ichimoku(high, low, close)
        result_df['Ichimoku_Tenkan'] = tenkan
        result_df['Ichimoku_Kijun'] = kijun
        result_df['Ichimoku_Senkou_A'] = senkou_a
        result_df['Ichimoku_Senkou_B'] = senkou_b
        result_df['Ichimoku_Chikou'] = chikou
        
        # Rate of Change
        result_df['ROC'] = calculate_roc(close)
        
        # CCI
        result_df['CCI'] = calculate_cci(high, low, close)
        
        # Volume indicators (if volume data is available)
        if 'volume' in company_data.columns and not company_data['volume'].isna().all():
            volume = company_data['volume']
            result_df['OBV'] = calculate_obv(close, volume)
            result_df['AD_Line'] = calculate_ad_line(high, low, close, volume)
            
            # Volume Moving Averages
            result_df['Volume_SMA_20'] = calculate_sma(volume, 20)
            result_df['Volume_Ratio'] = volume / result_df['Volume_SMA_20'].replace(0, np.nan)
            
            # Volume Profile (only if sufficient data)
            if len(close) > 50 and not volume.isna().all():
                vol_at_price, dominant_price = calculate_volume_profile(close, volume)
                result_df['Volume_At_Price'] = vol_at_price
                result_df['Dominant_Price_Level'] = dominant_price
            else:
                result_df['Volume_At_Price'] = np.nan
                result_df['Dominant_Price_Level'] = np.nan
        else:
            # Set volume indicators to NaN if no volume data
            result_df['OBV'] = np.nan
            result_df['AD_Line'] = np.nan
            result_df['Volume_SMA_20'] = np.nan
            result_df['Volume_Ratio'] = np.nan
            result_df['Volume_At_Price'] = np.nan
            result_df['Dominant_Price_Level'] = np.nan
        
        # Pivot Points
        pivot, r1, r2, r3, s1, s2, s3 = calculate_pivot_points(
            high.shift(1), low.shift(1), close.shift(1)
        )
        result_df['Pivot'] = pivot
        result_df['Resistance_1'] = r1
        result_df['Resistance_2'] = r2
        result_df['Resistance_3'] = r3
        result_df['Support_1'] = s1
        result_df['Support_2'] = s2
        result_df['Support_3'] = s3
        
        # Fibonacci Retracement Levels
        fib_236, fib_382, fib_500, fib_618, fib_786 = add_fibonacci_levels(high, low)
        result_df['Fib_23.6'] = fib_236
        result_df['Fib_38.2'] = fib_382
        result_df['Fib_50.0'] = fib_500
        result_df['Fib_61.8'] = fib_618
        result_df['Fib_78.6'] = fib_786
        
        # Additional Price-based indicators
        result_df['Price_Change'] = close.pct_change()
        result_df['Price_Change_abs'] = abs(result_df['Price_Change'])
        result_df['High_Low_Ratio'] = high / low
        result_df['Close_Open_Ratio'] = close / open_price
        
        # True Range
        result_df['True_Range'] = calculate_atr(high, low, close, window=1)
        
        # Technical Indicators - Additional
        result_df['TSI'] = calculate_tsi(close)
        result_df['MFI'] = calculate_mfi(high, low, close, volume)
        result_df['PVT'] = calculate_pvt(close, volume)
        result_df['TEMA'] = calculate_tema(close)
        result_df['KAMA'] = calculate_kama(close)
        result_df['Ulcer_Index'] = calculate_ulcer_index(close)
        
        return result_df
        
    except Exception as e:
        print(f"  Error processing company data: {e}")
        return None

def calculate_tsi(close, first_smooth=25, second_smooth=13):
    """True Strength Index (TSI)"""
    if len(close) < max(first_smooth, second_smooth) + 10:
        return pd.Series(index=close.index, dtype=float)
    
    # Calculate price momentum
    momentum = close.diff()
    
    # Double smoothing of momentum
    momentum_smoothed_1 = momentum.ewm(span=first_smooth).mean()
    momentum_smoothed_2 = momentum_smoothed_1.ewm(span=second_smooth).mean()
    
    # Double smoothing of absolute momentum
    abs_momentum = momentum.abs()
    abs_momentum_smoothed_1 = abs_momentum.ewm(span=first_smooth).mean()
    abs_momentum_smoothed_2 = abs_momentum_smoothed_1.ewm(span=second_smooth).mean()
    
    # Calculate TSI
    tsi = 100 * (momentum_smoothed_2 / abs_momentum_smoothed_2.replace(0, np.nan))
    return tsi

def calculate_mfi(high, low, close, volume, window=14):
    """Money Flow Index (MFI)"""
    if volume.isna().all() or len(close) < window + 5:
        return pd.Series(index=close.index, dtype=float)
    
    # Clean data
    clean_high = high.fillna(method='ffill').fillna(method='bfill')
    clean_low = low.fillna(method='ffill').fillna(method='bfill')
    clean_close = close.fillna(method='ffill').fillna(method='bfill')
    clean_volume = volume.fillna(0)
    
    # Calculate typical price
    typical_price = (clean_high + clean_low + clean_close) / 3
    
    # Calculate raw money flow
    raw_money_flow = typical_price * clean_volume
    
    # Calculate positive and negative money flow
    price_change = typical_price.diff()
    positive_flow = pd.Series(np.where(price_change > 0, raw_money_flow, 0), index=close.index)
    negative_flow = pd.Series(np.where(price_change < 0, raw_money_flow, 0), index=close.index)
    
    # Calculate MFI
    positive_mf = positive_flow.rolling(window=window).sum()
    negative_mf = negative_flow.rolling(window=window).sum()
    
    money_ratio = positive_mf / negative_mf.replace(0, np.nan)
    mfi = 100 - (100 / (1 + money_ratio))
    
    return mfi

def calculate_pvt(close, volume):
    """Price Volume Trend (PVT)"""
    if volume.isna().all() or close.isna().all():
        return pd.Series(index=close.index, dtype=float)
    
    # Clean data
    clean_close = close.fillna(method='ffill').fillna(method='bfill')
    clean_volume = volume.fillna(0)
    
    # Calculate price change percentage
    price_change_pct = clean_close.pct_change()
    
    # Calculate PVT
    pvt_change = price_change_pct * clean_volume
    pvt = pvt_change.cumsum()
    
    return pvt

def calculate_tema(data, window=14):
    """Triple Exponential Moving Average (TEMA)"""
    if len(data) < window * 3:
        return pd.Series(index=data.index, dtype=float)
    
    # First EMA
    ema1 = data.ewm(span=window).mean()
    
    # Second EMA (EMA of EMA1)
    ema2 = ema1.ewm(span=window).mean()
    
    # Third EMA (EMA of EMA2)
    ema3 = ema2.ewm(span=window).mean()
    
    # TEMA formula
    tema = 3 * ema1 - 3 * ema2 + ema3
    
    return tema

def calculate_kama(data, window=14, fast_sc=2, slow_sc=30):
    """Kaufman's Adaptive Moving Average (KAMA)"""
    if len(data) < window + 10:
        return pd.Series(index=data.index, dtype=float)
    
    # Calculate change and volatility
    change = abs(data - data.shift(window))
    volatility = data.diff().abs().rolling(window=window).sum()
    
    # Calculate efficiency ratio
    efficiency_ratio = change / volatility.replace(0, np.nan)
    
    # Calculate smoothing constant
    fast_sc_eff = 2.0 / (fast_sc + 1)
    slow_sc_eff = 2.0 / (slow_sc + 1)
    sc = (efficiency_ratio * (fast_sc_eff - slow_sc_eff) + slow_sc_eff) ** 2
    
    # Calculate KAMA
    kama = pd.Series(index=data.index, dtype=float)
    kama.iloc[window-1] = data.iloc[window-1]  # Initial value
    
    for i in range(window, len(data)):
        if not pd.isna(sc.iloc[i]):
            kama.iloc[i] = kama.iloc[i-1] + sc.iloc[i] * (data.iloc[i] - kama.iloc[i-1])
        else:
            kama.iloc[i] = kama.iloc[i-1]
    
    return kama

def calculate_ulcer_index(close, window=14):
    """Ulcer Index - Volatility measure focusing on downside risk"""
    if len(close) < window + 5:
        return pd.Series(index=close.index, dtype=float)
    
    # Calculate percentage drawdowns
    rolling_max = close.rolling(window=window).max()
    drawdowns = ((close - rolling_max) / rolling_max) * 100
    
    # Calculate squared drawdowns
    squared_drawdowns = drawdowns ** 2
    
    # Calculate Ulcer Index
    ulcer_index = np.sqrt(squared_drawdowns.rolling(window=window).mean())
    
    return ulcer_index

def generate_data_quality_report(df):
    """Generate a comprehensive data quality report"""
    print("\n" + "="*60)
    print("DATA QUALITY REPORT")
    print("="*60)
    
    # Overall statistics
    print(f"Total records: {len(df):,}")
    print(f"Total companies: {df['companyid'].nunique() if 'companyid' in df.columns else 'N/A'}")
    print(f"Date range: {df.index.min()} to {df.index.max()}" if hasattr(df.index, 'min') else "Date info not available")
    
    # Missing data analysis
    print(f"\nMISSING DATA ANALYSIS:")
    missing_summary = df.isnull().sum()
    missing_pct = (missing_summary / len(df)) * 100
    
    # Show columns with missing data
    cols_with_missing = missing_summary[missing_summary > 0].sort_values(ascending=False)
    if len(cols_with_missing) > 0:
        print(f"Columns with missing data:")
        for col, count in cols_with_missing.head(10).items():
            pct = missing_pct[col]
            print(f"  {col}: {count:,} ({pct:.2f}%)")
        if len(cols_with_missing) > 10:
            print(f"  ... and {len(cols_with_missing) - 10} more columns")
    else:
        print("  No missing data found!")
    
    # Technical indicators completeness
    print(f"\nTECHNICAL INDICATORS COMPLETENESS:")
    indicator_cols = [col for col in df.columns if col not in ['companyid', 'companyName', 'open', 'high', 'low', 'close', 'volume']]
    
    if indicator_cols:
        print(f"Total indicators calculated: {len(indicator_cols)}")
        
        # Check which indicators have the most complete data
        indicator_completeness = {}
        for col in indicator_cols:
            completeness = ((df[col].notna().sum() / len(df)) * 100)
            indicator_completeness[col] = completeness
        
        # Sort by completeness
        sorted_indicators = sorted(indicator_completeness.items(), key=lambda x: x[1], reverse=True)
        
        print(f"Top 10 most complete indicators:")
        for indicator, completeness in sorted_indicators[:10]:
            print(f"  {indicator}: {completeness:.1f}%")
        
        print(f"\nBottom 5 least complete indicators:")
        for indicator, completeness in sorted_indicators[-5:]:
            print(f"  {indicator}: {completeness:.1f}%")
    
    # Data range validation
    print(f"\nDATA VALIDATION SUMMARY:")
    if 'close' in df.columns:
        print(f"Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
        negative_prices = (df['close'] < 0).sum()
        if negative_prices > 0:
            print(f"WARNING: {negative_prices} negative close prices found")
    
    if 'volume' in df.columns:
        print(f"Volume range: {df['volume'].min():,.0f} - {df['volume'].max():,.0f}")
        zero_volume = (df['volume'] == 0).sum()
        print(f"Zero volume records: {zero_volume:,} ({(zero_volume/len(df)*100):.1f}%)")
    
    print("="*60)

if __name__ == "__main__":
    # File paths (relative to project root)
    input_file = "../data/raw/priceData5Year.csv"
    output_file = "../data/processed/stock_data_with_technical_indicators.csv"
    
    # For large datasets (>1M rows), use chunk processing
    # Set chunk_size=None to load entire dataset at once
    chunk_size = 500000  # Process 500k rows at a time
    
    print(f"Processing {input_file}...")
    print("Note: This is a large dataset. Processing may take several minutes...")
    
    # Create the technical indicators dataset
    enhanced_df = create_technical_indicators_dataset(input_file, chunk_size=chunk_size)
    
    if enhanced_df is not None:
        # Save the enhanced dataset
        print(f"Saving enhanced dataset to {output_file}...")
        enhanced_df.to_csv(output_file)
        
        # Generate data quality report first
        generate_data_quality_report(enhanced_df)
        
        # Save the enhanced dataset
        print(f"\nSaving enhanced dataset to {output_file}...")
        enhanced_df.to_csv(output_file)
        
        print("\nDataset Summary:")
        print(f"Total rows: {enhanced_df.shape[0]:,}")
        print(f"Total columns: {enhanced_df.shape[1]}")
        
        print("\nTechnical Indicators Added:")
        original_cols = ['companyid', 'companyName', 'open', 'high', 'low', 'close', 'volume']
        new_cols = [col for col in enhanced_df.columns if col not in original_cols]
        for i, col in enumerate(new_cols, 1):
            print(f"{i:2d}. {col}")
        
        print(f"\nEnhanced dataset saved successfully as '{output_file}'")
        
        # Show sample of the data (for first company if multi-company dataset)
        if 'companyid' in enhanced_df.columns:
            first_company = enhanced_df['companyid'].iloc[0]
            sample_data = enhanced_df[enhanced_df['companyid'] == first_company].head()
            print(f"\nSample data for company {first_company} (first 5 rows):")
            key_cols = ['open', 'high', 'low', 'close', 'RSI', 'MACD', 'BB_Position']
            available_cols = [col for col in key_cols if col in sample_data.columns]
            print(sample_data[available_cols].round(4).to_string())
        else:
            print("\nSample of the enhanced dataset (first 5 rows):")
            key_cols = ['open', 'high', 'low', 'close', 'RSI', 'MACD', 'BB_Position']
            available_cols = [col for col in key_cols if col in enhanced_df.columns]
            print(enhanced_df[available_cols].head().round(4).to_string())
        
        # Show basic statistics
        print("\nBasic statistics for key indicators:")
        key_indicators = ['RSI', 'MACD', 'BB_Position', 'ADX', 'ATR']
        available_indicators = [col for col in key_indicators if col in enhanced_df.columns]
        if available_indicators:
            stats_df = enhanced_df[available_indicators].describe()
            print(stats_df.round(4).to_string())
    else:
        print("Failed to create the technical indicators dataset.")
