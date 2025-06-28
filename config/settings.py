# Technical Indicators Configuration
# This file contains configurable parameters for the technical indicators calculation

# Data Processing Settings
CHUNK_SIZE = 500000  # Number of rows to process at once for large datasets
MIN_DATA_POINTS = 30  # Minimum number of data points required per company

# Moving Average Periods
SMA_PERIODS = [10, 20, 50]
EMA_PERIODS = [12, 26, 50]

# MACD Settings
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# RSI Settings
RSI_PERIOD = 14

# Bollinger Bands Settings
BB_PERIOD = 20
BB_STD_DEV = 2

# Stochastic Settings
STOCH_K_PERIOD = 14
STOCH_D_PERIOD = 3

# ATR Period
ATR_PERIOD = 14

# ADX Period
ADX_PERIOD = 14

# Parabolic SAR Settings
PSAR_AF_START = 0.02
PSAR_AF_MAX = 0.2

# Ichimoku Settings
ICHIMOKU_TENKAN = 9
ICHIMOKU_KIJUN = 26
ICHIMOKU_SENKOU_B = 52

# ROC Period
ROC_PERIOD = 12

# CCI Period
CCI_PERIOD = 20

# Volume Profile Settings
VOLUME_PROFILE_BINS = 20

# Fibonacci Lookback Period
FIBONACCI_LOOKBACK = 50

# File Paths
DATA_RAW_PATH = "data/raw/"
DATA_PROCESSED_PATH = "data/processed/"
OUTPUT_PATH = "output/"

# Column Mappings (for automatic detection)
PRICE_DATE_COLS = ['date', 'time', 'pricedate']
OPEN_COLS = ['open', 'openprice']
HIGH_COLS = ['high', 'adjustedhighprice', 'highprice']
LOW_COLS = ['low', 'adjustedlowprice', 'lowprice']
CLOSE_COLS = ['close', 'adjustedcloseprice', 'closeprice']
VOLUME_COLS = ['volume', 'quantity', 'traded', 'tradedquantity']

# Data Quality Thresholds
MAX_MISSING_PERCENTAGE = 50  # Skip companies if >50% of OHLC data is missing
MIN_ROWS_PER_COMPANY = 50   # Minimum rows required per company
