# LSTM Trading Strategy Configuration
# Customize these parameters for your trading strategy

# ===== MODEL PARAMETERS =====
SEQUENCE_LENGTH = 60          # Number of days to look back for LSTM input
LSTM_UNITS = [100, 50]       # LSTM layer sizes [first_layer, second_layer, ...]
DROPOUT_RATE = 0.2           # Dropout rate for regularization
EPOCHS = 50                  # Training epochs
BATCH_SIZE = 32              # Training batch size
VALIDATION_SPLIT = 0.1       # Fraction of training data for validation

# ===== TRADING PARAMETERS =====
TARGET_HORIZON = 5           # Days to hold position
TARGET_GAIN = 0.10          # Target profit (10% = 0.10)
STOP_LOSS = -0.03           # Stop loss (-3% = -0.03)
TEST_SPLIT = 0.2            # Fraction of data for testing

# ===== FEATURES TO USE =====
# Primary feature (required)
PRIMARY_FEATURE = 'close'

# Additional technical indicators to include (if available)
TECHNICAL_INDICATORS = [
    # Momentum Indicators (Priority)
    'RSI',              # Relative Strength Index
    'ROC',              # Rate of Change  
    'Stoch_K',          # Stochastic %K
    'Stoch_D',          # Stochastic %D
    'TSI',              # True Strength Index
    
    # Volume Indicators (Priority)
    'OBV',              # On Balance Volume
    'MFI',              # Money Flow Index
    'PVT',              # Price Volume Trend
    
    # Trend Indicators (Priority) 
    'MACD',             # MACD Line
    'TEMA',             # Triple Exponential Moving Average
    'KAMA',             # Kaufman's Adaptive Moving Average
    
    # Volatility Indicators (Priority)
    'ATR',              # Average True Range
    'BB_Position',      # Bollinger Band Position
    'Ulcer_Index',      # Ulcer Index
    
    # Additional Supporting Indicators
    'ADX',              # Average Directional Index
    'Volume_Ratio',     # Volume Ratio
    'Price_Change'      # Price Change
]

# ===== SIGNAL GENERATION =====
TREND_THRESHOLD = 0.001     # Minimum price change to consider as trend (0.1%)
CONFIDENCE_THRESHOLD = 0.6   # Minimum confidence for signal generation

# ===== RISK MANAGEMENT =====
MAX_POSITION_SIZE = 1.0     # Maximum position size (1.0 = 100% of capital)
RISK_FREE_RATE = 0.02       # Risk-free rate for Sharpe ratio calculation

# ===== OUTPUT SETTINGS =====
SAVE_MODEL = True           # Whether to save the trained model
SAVE_PLOTS = True           # Whether to save performance plots
PLOT_DPI = 300              # Plot resolution
VERBOSE_TRAINING = 1        # Training verbosity (0=silent, 1=progress bar, 2=epoch)

# ===== DATA REQUIREMENTS =====
MIN_DATA_POINTS = 1000      # Minimum data points required for training
MIN_TEST_SAMPLES = 100      # Minimum samples in test set

# ===== PERFORMANCE METRICS =====
METRICS_TO_TRACK = [
    'accuracy',             # Prediction accuracy
    'precision',            # Signal precision
    'recall',               # Signal recall
    'f1_score',            # F1 score
    'sharpe_ratio',        # Risk-adjusted returns
    'max_drawdown',        # Maximum drawdown
    'profit_factor',       # Profit factor
    'win_rate'             # Win rate percentage
]

# ===== ADVANCED SETTINGS =====
USE_EARLY_STOPPING = True   # Use early stopping during training
EARLY_STOPPING_PATIENCE = 10  # Epochs to wait before stopping
REDUCE_LR_PATIENCE = 5      # Epochs to wait before reducing learning rate
LEARNING_RATE_FACTOR = 0.5  # Factor to reduce learning rate

# ===== COMPANY FILTERING =====
# Set to None to use first company, or specify company ID
TARGET_COMPANY_ID = None

# Alternative: Use top N companies by data volume
USE_TOP_N_COMPANIES = None  # Set to integer to use top N companies

# ===== FILE PATHS =====
INPUT_DATA_PATH = "data/processed/stock_data_with_technical_indicators.csv"
OUTPUT_PATH = "output/"
MODEL_SAVE_PATH = "output/models/"

# Output file names
SIGNALS_OUTPUT_FILE = "lstm_trade_signals.csv"
SUMMARY_OUTPUT_FILE = "lstm_strategy_summary.csv"
REPORT_OUTPUT_FILE = "lstm_strategy_report.txt"
PLOTS_OUTPUT_FILE = "lstm_strategy_analysis.png"
MODEL_OUTPUT_FILE = "lstm_trading_model.h5"
