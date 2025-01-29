import pandas as pd
import plotly.graph_objects as go
import numpy as np

# ------------------------------------------------------------------------------
# 1. Load reference data and select EXACTLY 25 gold points
#    based on ascending 'Average Distance' in the reference CSV.
# ------------------------------------------------------------------------------
reference_df = pd.read_csv('significant_compounds_by_distance.csv')

# Ensure numeric
reference_df['Calc. MW'] = pd.to_numeric(reference_df['Calc. MW'], errors='coerce').fillna(0)

# Sort ascending by 'Average Distance'
reference_df = reference_df.sort_values('Average Distance')

# Shuffle reference entries to avoid positional bias
shuffled_reference = reference_df.sample(frac=1, random_state=42)
mw_values_shuffled = shuffled_reference['Calc. MW'].tolist()

# ------------------------------------------------------------------------------
# 2. Load main data
# ------------------------------------------------------------------------------
data = pd.read_csv('Compounds_1_27_25_Prepared.csv')

# Fill missing string columns with placeholders
data['Name'] = data['Name'].fillna('[-]')
data['Formula'] = data['Formula'].fillna('[-]')

for col in ['Calc. MW', 'm/z', 'RT [min]']:
    data[col] = data[col].fillna('').astype(str)

# ------------------------------------------------------------------------------
# 3. Validation & Cleaning Function
# ------------------------------------------------------------------------------
def validate_and_clean_column(df, column_name):
    """
    Ensures a column is numeric, with invalid strings replaced by NaN.
    Raises if >5% of values can't be converted to float.
    """
    try:
        if column_name not in df.columns:
            raise KeyError(f"Column '{column_name}' not found in the dataset.")
        
        # Strip and replace empty strings with '0'
        df[column_name] = df[column_name].astype(str).str.strip()
        df[column_name] = df[column_name].replace('', '0')
        
        # Check for non-numeric
        def is_invalid(val):
            try:
                float(val)
                return False
            except ValueError:
                return True
         
        invalid_mask = df[column_name].apply(is_invalid)
        invalid_percentage = invalid_mask.mean() * 100

        if invalid_percentage > 5:
            # Show sample of problematic entries
            problematic_entries = df[invalid_mask]
            sample_indices = problematic_entries.sample(
                min(10, len(problematic_entries))
            ).index
            
            print(f"\nProblematic entries in '{column_name}':")
            for idx in sample_indices:
                original_value = df.loc[idx, column_name]
                print(f"Index {idx}: '{original_value}'")
            
            raise ValueError(
                f"More than 5% ({invalid_percentage:.2f}%) of values in column '{column_name}' are invalid."
            )
        
        print(f"Column '{column_name}' cleaned successfully with {invalid_percentage:.2f}% invalid values.")

        # Convert to numeric with errors coerced to NaN
        numeric_values = pd.to_numeric(df[column_name], errors='coerce')
        return numeric_values

    except Exception as e:
        print(f"Error validating column '{column_name}': {e}")
        raise

# ------------------------------------------------------------------------------
# 4. Configuration for multiple comparisons
# ------------------------------------------------------------------------------
config = {
    "(cla_and_lps) / (lps)": {
        'ratio_col': "Ratio: (cla_and_lps) / (lps)",
        'p_value_col': "P-value: (cla_and_lps) / (lps)",
        'title': "Volcano Plot: (cla_and_lps) / (lps)"
    },
    "(no2-cla_and_lps) / (lps)": {
        'ratio_col': "Ratio: (no2-cla_and_lps) / (lps)",
        'p_value_col': "P-value: (no2-cla_and_lps) / (lps)",
        'title': "Volcano Plot: (no2-cla_and_lps) / (lps)"
    }
}

# ------------------------------------------------------------------------------
# 5. Identify EXACTLY 25 gold indices in the main data
#    by walking down 'Average Distance' from the reference.
# ------------------------------------------------------------------------------
GOLD_LIMIT = 30  # Set to 25 as per original functionality
tolerance = 1e-6  # Tolerance for Calc. MW matching

mw_values_sorted = mw_values_shuffled  # Use shuffled reference
gold_indices = []
mw_idx = 0

# Convert main 'Calc. MW' to numeric once, so we don't do it repeatedly
data_mw = pd.to_numeric(data['Calc. MW'], errors='coerce')

while len(gold_indices) < GOLD_LIMIT and mw_idx < len(mw_values_sorted):
    candidate_mw = mw_values_sorted[mw_idx]
    # Find matches within tolerance of candidate_mw
    matching_rows = data.index[
        np.isclose(data_mw, candidate_mw, atol=tolerance, rtol=0, equal_nan=False)
    ].tolist()

    for row_idx in matching_rows:
        if len(gold_indices) >= GOLD_LIMIT:
            break
        gold_indices.append(row_idx)

    mw_idx += 1

# Debugging logs
print(f"Total gold indices selected: {len(gold_indices)}")
print(f"Unique matched Calc. MW values: {len(set(data_mw[gold_indices]))}")

gold_indices_set = set(gold_indices)  # for quick membership checks

# ------------------------------------------------------------------------------
# 6. Build a hover text list for each row in 'data'
# ------------------------------------------------------------------------------
hover_text = data.apply(lambda row: (
    f"Name:\t{row['Name']}<br>"
    f"Formula:\t{row['Formula']}<br>"
    f"Calc. MW:\t{row['Calc. MW']}<br>"
    f"m/z:\t{row['m/z']}<br>"
    f"RT [min]:\t{row['RT [min]']}"
), axis=1).tolist()

# ------------------------------------------------------------------------------
# 7. Generate traces for each comparison
# ------------------------------------------------------------------------------
traces = []
first_comparison = next(iter(config))  # The first one visible by default

# We will also collect the global min and max log2(fold change) for auto-zoom
all_fc_values = []

for comp_idx, (comp, params) in enumerate(config.items()):
    # Validate the ratio & p-value columns
    try:
        ratio = validate_and_clean_column(data, params['ratio_col'])
        pvals = validate_and_clean_column(data, params['p_value_col'])
    except Exception as e:
        print(f"Skipped comparison '{comp}' due to error: {e}")
        continue

    # Convert to safe ratio/pval
    safe_ratio = ratio.replace(0, np.nan)  # avoid -inf in log2
    safe_pvals = pvals.replace(0, np.nan)  # avoid -inf in -log10

    log2_ratio = np.log2(safe_ratio)
    neg_log_p = -np.log10(safe_pvals)

    # Add these log2 values to global list (for auto x-axis range)
    all_fc_values.extend(log2_ratio.dropna().values)

    # Lists for non-gold vs gold
    non_gold_x = []
    non_gold_y = []
    non_gold_colors = []
    non_gold_hover = []

    gold_x = []
    gold_y = []
    gold_hover = []

    # Single pass for alignment
    for idx in data.index:
        fc = log2_ratio[idx]
        pval = neg_log_p[idx]
        point_hover = (
            f"{hover_text[idx]}<br>"
            f"log2(Fold Change): {fc:.3f}<br>"
            f"-log10(P-value): {pval:.3f}"
        )

        if idx in gold_indices_set:
            # This is one of the EXACT 25 gold points
            gold_x.append(fc)
            gold_y.append(pval)
            gold_hover.append(point_hover)
        else:
            # Determine color based on thresholds if valid fc/pval
            if not pd.isna(fc) and not pd.isna(pval):
                if fc > 0.5 and pval > -np.log10(0.05):
                    color = 'green'
                elif fc < -0.5 and pval > -np.log10(0.05):
                    color = 'red'
                else:
                    color = 'blue'
            else:
                # missing ratio or p-value
                color = 'blue'

            non_gold_x.append(fc)
            non_gold_y.append(pval)
            non_gold_colors.append(color)
            non_gold_hover.append(point_hover)

    # Create non-gold trace (becomes visible for its comparison)
    traces.append(go.Scatter(
        x=non_gold_x,
        y=non_gold_y,
        mode='markers',
        marker=dict(color=non_gold_colors, opacity=0.9),
        hovertext=non_gold_hover,
        hoverinfo='text',
        visible=(comp == first_comparison),
        showlegend=False,
        hoverlabel=dict(
            bgcolor='red',
            bordercolor='black',
            font=dict(size=16, family='Arial')
        )
    ))

    # Create gold trace (plotted on top, same visibility)
    traces.append(go.Scatter(
        x=gold_x,
        y=gold_y,
        mode='markers',
        marker=dict(color='gold', opacity=1),
        hovertext=gold_hover,
        hoverinfo='text',
        visible=(comp == first_comparison),
        showlegend=False,
        hoverlabel=dict(
            bgcolor='gold',
            bordercolor='black',
            font=dict(size=16, family='Arial')
        )
    ))

# ------------------------------------------------------------------------------
# 8. Legend (added at the end)
# ------------------------------------------------------------------------------
legend_traces = [
    go.Scatter(x=[None], y=[None], mode='markers',
               marker=dict(color='gold'), name='Most Significant by Average Distance',
               showlegend=True),
    go.Scatter(x=[None], y=[None], mode='markers',
               marker=dict(color='green'), name='Significant Upregulated',
               showlegend=True),
    go.Scatter(x=[None], y=[None], mode='markers',
               marker=dict(color='red'), name='Significant Downregulated',
               showlegend=True),
    go.Scatter(x=[None], y=[None], mode='markers',
               marker=dict(color='blue'), name='Insignificant',
               showlegend=True)
]

all_traces = traces + legend_traces

# ------------------------------------------------------------------------------
# 9. Dropdown Menu for Comparison Switching
# ------------------------------------------------------------------------------
num_comparisons = len(config)
buttons = []
for i, (comp, params) in enumerate(config.items()):
    # Each comparison has 2 traces (non-gold + gold), then 4 legend traces
    visible = [False] * (2 * num_comparisons)
    visible[2*i] = True
    visible[2*i + 1] = True
    visible += [True] * len(legend_traces)  # keep legend always visible

    buttons.append(dict(
        label=params['title'],
        method='update',
        args=[
            {'visible': visible},
            {'title': params['title']}
        ]
    ))

# ------------------------------------------------------------------------------
# 10. Threshold Lines
# ------------------------------------------------------------------------------
threshold_lines = [
    dict(type='line', x0=-10, x1=10, y0=-np.log10(0.05), y1=-np.log10(0.05),
         line=dict(color='black', dash='dash')),
    dict(type='line', x0=-0.5, x1=-0.5, y0=0, y1=10,
         line=dict(color='black', dash='dash')),
    dict(type='line', x0=0.5, x1=0.5, y0=0, y1=10,
         line=dict(color='black', dash='dash'))
]

# ------------------------------------------------------------------------------
# 11. Create and Configure Figure
# ------------------------------------------------------------------------------
fig = go.Figure(data=all_traces)

# Default to first comparison's title
fig.update_layout(
    title=config[first_comparison]['title'],
    xaxis=dict(title='log2(Fold Change)', range=[-6, 5], gridcolor='gray'),  # Hardcoded x-axis range
    yaxis=dict(title='-log10(p-value)', range=[-0.2, 5], gridcolor='gray'),
    plot_bgcolor='lightslategray',
    paper_bgcolor='lightslategray',
    font=dict(color='black'),
    shapes=threshold_lines,
    showlegend=True,
    updatemenus=[dict(
        type='dropdown',
        x=1.15,
        y=1.15,
        showactive=True,
        buttons=buttons,
        direction='down',
        pad=dict(r=10, t=10)
    )]
)

# ------------------------------------------------------------------------------
# 12. Save Output
# ------------------------------------------------------------------------------
fig.write_html('volcano_plot.html')