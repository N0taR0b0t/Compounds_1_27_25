import pandas as pd
import numpy as np

# Function to calculate distances
def calculate_distance_to_extremes(data, log2_fc_cols, p_value_cols, distance_cap=None):
    results = []

    for idx, compound in data.iterrows():
        total_distance = 0
        comparison_count = 0

        for log2_fc_col, p_value_col in zip(log2_fc_cols, p_value_cols):
            # Get the current compound's values
            x = compound[log2_fc_col]
            y = compound[p_value_col]

            # Ensure valid numeric values
            if pd.isna(x) or pd.isna(y) or not isinstance(y, (int, float)) or not isinstance(x, (int, float)):
                continue

            # Calculate -log10(P-value) and ensure it's valid
            y = -np.log10(y) if y > 0 else np.nan
            if pd.isna(y) or np.isinf(y):
                continue

            # Identify the region of interest
            comparison_data = data[[log2_fc_col, p_value_col]].copy()
            comparison_data["-log10(P-value)"] = -np.log10(comparison_data[p_value_col].replace(0, np.nan))

            if x < 0:
                # Left region (x < 0)
                leftmost_x = comparison_data[log2_fc_col][comparison_data[log2_fc_col] < 0].min()
                topmost_y = comparison_data["-log10(P-value)"][comparison_data[log2_fc_col] < 0].max()
                if not pd.isna(leftmost_x) and not pd.isna(topmost_y):
                    distance = np.sqrt((x - leftmost_x) ** 2 + (y - topmost_y) ** 2)
                    total_distance += min(distance, distance_cap) if distance_cap else distance
                    comparison_count += 1
            elif x >= 0:
                # Right region (x >= 0)
                rightmost_x = comparison_data[log2_fc_col][comparison_data[log2_fc_col] >= 0].max()
                topmost_y = comparison_data["-log10(P-value)"][comparison_data[log2_fc_col] >= 0].max()
                if not pd.isna(rightmost_x) and not pd.isna(topmost_y):
                    distance = np.sqrt((x - rightmost_x) ** 2 + (y - topmost_y) ** 2)
                    total_distance += min(distance, distance_cap) if distance_cap else distance
                    comparison_count += 1

        # Calculate average distance
        avg_distance = total_distance / comparison_count if comparison_count > 0 else np.nan

        # Use only "Calc. MW" as the identifier
        identifier = compound["Calc. MW"]
        results.append({"Calc. MW": identifier, "Average Distance": avg_distance})

    return pd.DataFrame(results)

# Load your data
input_file = "Compounds_1_27_25_Prepared.csv"
output_file = "significant_compounds_by_distance.csv"  # Replace with desired output file path
data = pd.read_csv(input_file)

# Ensure all relevant columns are numeric
data = data.applymap(lambda x: pd.to_numeric(x, errors='coerce') if isinstance(x, str) else x)

# Define relevant columns for log2 fold changes and p-values
log2_fc_cols = [col for col in data.columns if "Log2 Fold Change" in col]
p_value_cols = [col for col in data.columns if "P-value" in col and "Adj." not in col]

# Perform the refined distance calculation
results = calculate_distance_to_extremes(data, log2_fc_cols, p_value_cols)

# Drop NaN values and sort by ascending average distance
results.dropna(subset=["Average Distance"], inplace=True)
results.sort_values(by="Average Distance", inplace=True)

# Save results to CSV
results.to_csv(output_file, index=False)
print(f"Results saved to {output_file}")