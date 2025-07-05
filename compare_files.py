import pandas as pd
import os

# File paths
file1 = 'C:/Users/98765/OneDrive/Desktop/Timesnow/src/processed_data/enhanced_priceData5Year.csv'
file2 = 'C:/Users/98765/OneDrive/Desktop/Timesnow/stock_data_with_technical_indicators.csv'

print('=== COMPREHENSIVE CSV FILE COMPARISON REPORT ===')
print()

# Basic file info
size1 = os.path.getsize(file1)
size2 = os.path.getsize(file2)

print(f'FILE 1: enhanced_priceData5Year.csv')
print(f'  Location: {file1}')
print(f'  Size: {size1:,} bytes ({size1/1024/1024/1024:.2f} GB)')

print(f'\nFILE 2: stock_data_with_technical_indicators.csv')
print(f'  Location: {file2}')
print(f'  Size: {size2:,} bytes ({size2/1024/1024/1024:.2f} GB)')

print(f'\nSIZE DIFFERENCE: {abs(size2-size1):,} bytes')
print(f'File 2 is {size2/size1:.2f}x larger than File 1')

# Read headers and compare columns
with open(file1, 'r') as f:
    header1 = f.readline().strip()
with open(file2, 'r') as f:
    header2 = f.readline().strip()

cols1 = [col.strip() for col in header1.split(',')]
cols2 = [col.strip() for col in header2.split(',')]

print(f'\nCOLUMN ANALYSIS:')
print(f'File 1 columns: {len(cols1)}')
print(f'File 2 columns: {len(cols2)}')
print(f'Additional columns in File 2: {len(cols2) - len(cols1)}')

# Find column differences
common_cols = set(cols1) & set(cols2)
only_in_file1 = set(cols1) - set(cols2)
only_in_file2 = set(cols2) - set(cols1)

print(f'\nCOLUMN BREAKDOWN:')
print(f'Common columns: {len(common_cols)}')
print(f'Unique to File 1: {len(only_in_file1)}')
print(f'Unique to File 2: {len(only_in_file2)}')

if only_in_file1:
    print(f'\nCOLUMNS ONLY IN FILE 1:')
    for col in sorted(only_in_file1):
        print(f'  • {col}')

if only_in_file2:
    print(f'\nCOLUMNS ONLY IN FILE 2:')
    for col in sorted(only_in_file2):
        print(f'  • {col}')

# Read samples for data comparison
print(f'\nDATA SAMPLE ANALYSIS:')
try:
    df1_sample = pd.read_csv(file1, nrows=10)
    df2_sample = pd.read_csv(file2, nrows=10)
    
    print(f'File 1 sample shape: {df1_sample.shape}')
    print(f'File 2 sample shape: {df2_sample.shape}')
    
    # Check data types
    print(f'\nDATA TYPES COMPARISON:')
    print(f'File 1 numeric columns: {len(df1_sample.select_dtypes(include=["number"]).columns)}')
    print(f'File 2 numeric columns: {len(df2_sample.select_dtypes(include=["number"]).columns)}')
    
    # Check for missing values
    print(f'\nMISSING VALUES IN SAMPLES:')
    print(f'File 1: {df1_sample.isnull().sum().sum()} total nulls')
    print(f'File 2: {df2_sample.isnull().sum().sum()} total nulls')
    
    # Company information
    company_col1 = None
    company_col2 = None
    
    if 'companyname' in df1_sample.columns:
        company_col1 = 'companyname'
    elif 'companyName' in df1_sample.columns:
        company_col1 = 'companyName'
    
    if 'companyname' in df2_sample.columns:
        company_col2 = 'companyname'
    elif 'companyName' in df2_sample.columns:
        company_col2 = 'companyName'
    
    print(f'\nCOMPANY DATA:')
    if company_col1:
        print(f'File 1 companies in sample: {df1_sample[company_col1].unique()[:5]}')
    if company_col2:
        print(f'File 2 companies in sample: {df2_sample[company_col2].unique()[:5]}')
    
except Exception as e:
    print(f'Error reading sample data: {e}')

# Estimate row counts
def estimate_row_count(filepath):
    size = os.path.getsize(filepath)
    with open(filepath, 'r') as f:
        lines = [f.readline() for _ in range(100)]
        avg_line_length = sum(len(line) for line in lines) / len(lines)
    return int(size / avg_line_length), avg_line_length

print(f'\nROW COUNT ESTIMATION:')
rows1, avg_len1 = estimate_row_count(file1)
rows2, avg_len2 = estimate_row_count(file2)

print(f'File 1 estimated rows: {rows1:,}')
print(f'File 2 estimated rows: {rows2:,}')
print(f'Row difference: {abs(rows2 - rows1):,}')
print(f'Average line length File 1: {avg_len1:.1f} characters')
print(f'Average line length File 2: {avg_len2:.1f} characters')

print(f'\n=== SUMMARY ===')
print(f'• File 2 has {len(cols2) - len(cols1)} additional columns')
print(f'• File 2 has approximately {rows2 - rows1:,} more rows')
print(f'• File 2 is {((size2/size1-1)*100):.1f}% larger in size')
print(f'• Both files appear to contain stock price data with technical indicators')
print(f'• File 2 includes more advanced technical indicators like Fibonacci levels, Ichimoku, etc.')
