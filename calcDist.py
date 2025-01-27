import pandas as pd
import numpy as np
from collections import defaultdict
import glob
import sys
import os

# Define required ratio columns
RATIO_COLUMNS = [
    "Ratio: (cla) / (ctrl)",
    "Ratio: (cla_and_lps) / (ctrl)",
    "Ratio: (cla_and_no2_and_lps) / (ctrl)",
    "Ratio: (lps) / (ctrl)",
    "Ratio: (no2-cla_and_lps) / (ctrl)",
    "Ratio: (cla) / (lps)",
    "Ratio: (cla_and_lps) / (lps)",
    "Ratio: (cla_and_no2_and_lps) / (lps)",
    "Ratio: (no2-cla_and_lps) / (lps)"
]

def main():
    """
    Reads CSV files matching 'Compounds_1_27_25_Prepared.csv',
    checks for required columns, converts data to numeric as needed,
    calculates Euclidean distance to a reference point for each compound,
    and outputs the top 50 by smallest distance.
    """
    # Track distances across files
    total_distances = defaultdict(float)
    
    # Find matching CSV files
    csv_files = glob.glob('Compounds_1_27_25_Prepared.csv')
    
    if not csv_files:
        print("No CSV files found matching 'Compounds_1_27_25_Prepared.csv'. Exiting.", file=sys.stderr)
        sys.exit(1)

    print("Found the following files to process:")
    for f in csv_files:
        print(f"  - {f}")

    # Process each CSV file
    for csv_file in csv_files:
        if not os.path.isfile(csv_file):
            print(f"File {csv_file} does not exist. Skipping...", file=sys.stderr)
            continue

        print(f"\nProcessing '{csv_file}'...")

        # Load data with proper encoding
        try:
            df = pd.read_csv(csv_file, encoding='ISO-8859-1')
        except Exception as e:
            print(f"  ERROR: Could not read {csv_file}. Reason: {e}", file=sys.stderr)
            continue

        # Validate required columns
        required_cols = ['Compounds ID', 'Name', 'Calc. MW'] + RATIO_COLUMNS
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"  ERROR: Missing columns in {csv_file}: {missing_cols}", file=sys.stderr)
            continue

        # Convert ratio columns to numeric (coerce => non-convertible becomes NaN)
        for ratio_col in RATIO_COLUMNS:
            if ratio_col in df.columns:
                df[ratio_col] = pd.to_numeric(df[ratio_col], errors='coerce')

        # For each ratio column, find the corresponding p-value column
        for ratio_col in RATIO_COLUMNS:
            pvalue_col = ratio_col.replace("Ratio: ", "P-value: ", 1)

            # If p-value column is missing, skip
            if pvalue_col not in df.columns:
                print(f"  Skipping ratio column '{ratio_col}' - no matching p-value column '{pvalue_col}'.")
                continue

            # Convert p-values to numeric
            df[pvalue_col] = pd.to_numeric(df[pvalue_col], errors='coerce')

            # Drop rows where either ratio or p-value is NaN
            df_temp = df.dropna(subset=[ratio_col, pvalue_col]).copy()
            if df_temp.empty:
                print(f"  WARNING: All rows are invalid or NaN for '{ratio_col}' or '{pvalue_col}'. Skipping.")
                continue
            
            # Show how many rows remain valid
            print(f"  Processing '{ratio_col}' vs '{pvalue_col}': {len(df_temp)} valid rows remain.")

            # Calculate -log10(p-value)
            try:
                df_temp['-log10_pvalue'] = -np.log10(df_temp[pvalue_col])
            except (ValueError, TypeError) as e:
                print(f"  ERROR: Could not calculate -log10 for column {pvalue_col}. Reason: {e}", file=sys.stderr)
                continue

            # Calculate reference points for the ratio
            leftmost = df_temp[ratio_col].min()
            rightmost = df_temp[ratio_col].max()
            topmost = df_temp['-log10_pvalue'].max()

            # For each valid compound, compute the distance
            for _, row in df_temp.iterrows():
                compound_id = row['Compounds ID']
                log2fc = row[ratio_col]
                y_val = row['-log10_pvalue']

                # Decide reference point (leftmost or rightmost)
                if log2fc < 0:
                    ref_x, ref_y = leftmost, topmost
                else:
                    ref_x, ref_y = rightmost, topmost

                # Euclidean distance
                distance = np.sqrt((log2fc - ref_x)**2 + (y_val - ref_y)**2)
                total_distances[compound_id] += distance

    # If no distances were computed, exit early
    if not total_distances:
        print("\nNo valid distances were computed. Please check your input data.", file=sys.stderr)
        sys.exit(1)

    # Create a dataframe of total distances
    distance_df = pd.DataFrame(
        total_distances.items(),
        columns=['Compounds ID', 'Total Distance']
    )
    print(f"\nComputed total distances for {len(distance_df)} compounds.")

    # Collect compound metadata from all CSV files
    compound_info_list = []
    for csv_file in csv_files:
        try:
            df_info = pd.read_csv(
                csv_file,
                usecols=['Compounds ID', 'Name', 'Calc. MW'],
                encoding='ISO-8859-1'
            )
            compound_info_list.append(df_info)
        except ValueError:
            # If the columns aren't present, skip
            print(f"  WARNING: Could not read compound metadata from {csv_file}. Missing columns.")
        except Exception as e:
            print(f"  WARNING: Error reading {csv_file} for compound metadata: {e}")

    # Merge and deduplicate compound info
    if compound_info_list:
        compound_info = pd.concat(compound_info_list).drop_duplicates('Compounds ID')
        # If 'Name' is missing, fill with 'Calc. MW' as fallback
        compound_info['Name'] = compound_info['Name'].fillna(compound_info['Calc. MW'])
    else:
        print("WARNING: No compound info collected. Output will contain only 'Compounds ID' and 'Total Distance'.")
        compound_info = pd.DataFrame(columns=['Compounds ID', 'Name', 'Calc. MW'])

    # Merge distances with compound info
    final_df = pd.merge(distance_df, compound_info, on='Compounds ID', how='left')

    # Sort by ascending distance and pick top 50
    final_df = final_df.sort_values('Total Distance', ascending=True).head(50)

    # Output results
    output_file = 'significant_compounds_by_distance.csv'
    final_df.to_csv(output_file, index=False)
    print(f"\nResults saved to '{output_file}'.\n")
    print("Top 50 compounds (Name | Total Distance):")
    print(final_df[['Name', 'Total Distance']].to_string(index=False))

if __name__ == '__main__':
    main()