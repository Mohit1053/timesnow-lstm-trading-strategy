"""
Technical Indicators Runner Script

This script runs the technical indicators calculation from the project root directory.
Usage: python run_analysis.py
"""

import os
import sys

# Add project root and src directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'src'))

# Change to project root directory
os.chdir(project_root)

# Import and run the main script
from technical_indicators import create_technical_indicators_dataset, generate_data_quality_report

if __name__ == "__main__":
    print("=" * 60)
    print("TECHNICAL INDICATORS ANALYSIS")
    print("=" * 60)
    
    # File paths
    input_file = "data/raw/priceData5Year.csv"
    output_file = "data/processed/stock_data_with_technical_indicators.csv"
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"ERROR: Input file not found: {input_file}")
        print("Please ensure the data file is in the data/raw/ directory")
        sys.exit(1)
    
    # Processing parameters
    chunk_size = 500000  # Process 500k rows at a time for large datasets
    
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print(f"Chunk size: {chunk_size:,} rows")
    print("Note: This is a large dataset. Processing may take several minutes...")
    print()
    
    # Create the technical indicators dataset
    enhanced_df = create_technical_indicators_dataset(input_file, chunk_size=chunk_size)
    
    if enhanced_df is not None:
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
        
        print(f"\n✅ Enhanced dataset saved successfully as '{output_file}'")
        
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
            
        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
    else:
        print("❌ Failed to create the technical indicators dataset.")
        sys.exit(1)
