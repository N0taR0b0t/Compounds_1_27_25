import pandas as pd
import plotly.graph_objects as go
import numpy as np

# Load reference data
reference_df = pd.read_csv('significant_compounds_by_distance.csv')
significant_compounds = reference_df['Name'].dropna().unique().tolist()

# Load and prepare main data
data = pd.read_csv('Compounds_1_27_25_Prepared.csv')

# Handle missing values
data['Name'] = data['Name'].fillna('[-]')
data['Formula'] = data['Formula'].fillna('[-]')
for col in ['Calc. MW', 'm/z', 'RT [min]']:
    data[col] = data[col].fillna('').astype(str)

# Function to validate and clean columns
def validate_and_clean_column(df, column_name):
    try:
        if column_name not in df.columns:
            raise KeyError(f"Column '{column_name}' not found in the dataset.")
        
        # Convert column to string, strip whitespace, replace empty strings with '0'
        df[column_name] = df[column_name].astype(str).str.strip()
        df[column_name] = df[column_name].replace('', '0')

        # Identify invalid (non-numeric) values (ignoring the ones that are now '0')
        def is_invalid(val):
            try:
                float(val)
                return False
            except ValueError:
                return True
        
        invalid_mask = df[column_name].apply(is_invalid)
        invalid_percentage = invalid_mask.mean() * 100

        if invalid_percentage > 5:
            problematic_entries = df[invalid_mask]
            problematic_sample = problematic_entries.sample(
                min(10, len(problematic_entries))
            ).index
            
            print(f"\nProblematic entries in '{column_name}':")
            for idx in problematic_sample:
                original_value = df.loc[idx, column_name]
                print(f"Index {idx}: '{original_value}'")
            
            raise ValueError(
                f"More than 5% ({invalid_percentage:.2f}%) of values in column '{column_name}' are invalid."
            )
        
        print(f"Column '{column_name}' cleaned successfully with {invalid_percentage:.2f}% invalid values.")

        # Convert column to numeric, coerce errors to NaN (NaN is ignored in the script output)
        numeric_values = pd.to_numeric(df[column_name], errors='coerce')
        return numeric_values

    except Exception as e:
        print(f"Error validating column '{column_name}': {e}")
        raise

# Configuration dictionary
config = {
    "CLA vs CTRL": {
        'ratio_col': "Ratio: (cla) / (ctrl)",
        'p_value_col': "P-value: (cla) / (ctrl)",
        'title': "Volcano Plot: CLA vs CTRL"
    },
    "CLA+LPS vs CTRL": {
        'ratio_col': "Ratio: (cla_and_lps) / (ctrl)",
        'p_value_col': "P-value: (cla_and_lps) / (ctrl)",
        'title': "Volcano Plot: CLA+LPS vs CTRL"
    },
    "CLA+NO2+LPS vs CTRL": {
        'ratio_col': "Ratio: (cla_and_no2_and_lps) / (ctrl)",
        'p_value_col': "P-value: (cla_and_no2_and_lps) / (ctrl)",
        'title': "Volcano Plot: CLA+NO2+LPS vs CTRL"
    },
    "LPS vs CTRL": {
        'ratio_col': "Ratio: (lps) / (ctrl)",
        'p_value_col': "P-value: (lps) / (ctrl)",
        'title': "Volcano Plot: LPS vs CTRL"
    },
    "NO2-CLA+LPS vs CTRL": {
        'ratio_col': "Ratio: (no2-cla_and_lps) / (ctrl)",
        'p_value_col': "P-value: (no2-cla_and_lps) / (ctrl)",
        'title': "Volcano Plot: NO2-CLA+LPS vs CTRL"
    },
    "CLA vs LPS": {
        'ratio_col': "Ratio: (cla) / (lps)",
        'p_value_col': "P-value: (cla) / (lps)",
        'title': "Volcano Plot: CLA vs LPS"
    },
    "CLA+LPS vs LPS": {
        'ratio_col': "Ratio: (cla_and_lps) / (lps)",
        'p_value_col': "P-value: (cla_and_lps) / (lps)",
        'title': "Volcano Plot: CLA+LPS vs LPS"
    },
    "CLA+NO2+LPS vs LPS": {
        'ratio_col': "Ratio: (cla_and_no2_and_lps) / (lps)",
        'p_value_col': "P-value: (cla_and_no2_and_lps) / (lps)",
        'title': "Volcano Plot: CLA+NO2+LPS vs LPS"
    },
    "NO2-CLA+LPS vs LPS": {
        'ratio_col': "Ratio: (no2-cla_and_lps) / (lps)",
        'p_value_col': "P-value: (no2-cla_and_lps) / (lps)",
        'title': "Volcano Plot: NO2-CLA+LPS vs LPS"
    }
}

# Create hover text
hover_text = data.apply(lambda row: (
    f"Name:\t{row['Name']}<br>"
    f"Formula:\t{row['Formula']}<br>"
    f"Calc. MW:\t{row['Calc. MW']}<br>"
    f"m/z:\t{row['m/z']}<br>"
    f"RT [min]:\t{row['RT [min]']}"
), axis=1).tolist()

# Create traces for each comparison
# Create traces for each comparison
traces = []
gold_traces = []  # Separate list for significant compounds
first_comparison = next(iter(config))

for comp, params in config.items():
    # Validate and clean ratio and p-value columns
    try:
        ratio = validate_and_clean_column(data, params['ratio_col'])
        pvals = validate_and_clean_column(data, params['p_value_col'])
    except Exception as e:
        print(f"Failed to process comparison '{comp}': {e}")
        continue

    # Log2 and -log10 transformations with fallback
    try:
        log2_ratio = np.log2(ratio.replace(0, np.nan))  # Replace zero with NaN to avoid -inf
        neg_log_p = -np.log10(pvals.replace(0, np.nan))
    except Exception as e:
        print(f"Error during transformations for '{comp}': {e}")
        continue

    # Assign colors and separate gold points
    colors = []
    gold_x = []
    gold_y = []
    gold_hover = []

    for idx in data.index:
        name = data.loc[idx, 'Name']
        fc = log2_ratio[idx]
        pval = neg_log_p[idx]

        if name in significant_compounds:
            gold_x.append(fc)
            gold_y.append(pval)
            gold_hover.append(hover_text[idx])
        elif not pd.isna(fc) and not pd.isna(pval):
            if fc > 0.5 and pval > -np.log10(0.05):
                colors.append('green')
            elif fc < -0.5 and pval > -np.log10(0.05):
                colors.append('red')
            else:
                colors.append('blue')
        else:
            colors.append('blue')

    # Append non-gold points trace
    traces.append(go.Scatter(
        x=log2_ratio[~data['Name'].isin(significant_compounds)],
        y=neg_log_p[~data['Name'].isin(significant_compounds)],
        mode='markers',
        marker=dict(color=colors, opacity=0.9),
        hovertext=[hover_text[i] for i in data.index if data.loc[i, 'Name'] not in significant_compounds],
        hoverinfo='text',
        visible=(comp == first_comparison),
        showlegend=False,
        hoverlabel=dict(
            bgcolor='red',
            bordercolor='black',
            font=dict(size=16, family='Arial')
        )
    ))

    # Append gold points trace
    gold_traces.append(go.Scatter(
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

# Add gold traces after all others
traces.extend(gold_traces)

# Create legend traces
legend_traces = [
    go.Scatter(x=[None], y=[None], mode='markers',
               marker=dict(color='gold'), name='Significant by Distance'),
    go.Scatter(x=[None], y=[None], mode='markers',
               marker=dict(color='green'), name='Significant Upregulated'),
    go.Scatter(x=[None], y=[None], mode='markers',
               marker=dict(color='red'), name='Significant Downregulated'),
    go.Scatter(x=[None], y=[None], mode='markers',
               marker=dict(color='blue'), name='Insignificant')
]

# Create dropdown buttons
buttons = []
for i, comp in enumerate(config):
    visible = [False] * len(config)
    visible[i] = True
    # The last 4 'True's make the legend traces always visible
    buttons.append(dict(
        label=config[comp]['title'],
        method='update',
        args=[{'visible': visible + [True]*4},
              {'title': config[comp]['title']}]
    ))

# Create threshold lines
threshold_lines = [
    dict(type='line', x0=-10, x1=10, y0=-np.log10(0.05), y1=-np.log10(0.05),
         line=dict(color='black', dash='dash')),
    dict(type='line', x0=-0.5, x1=-0.5, y0=0, y1=10,
         line=dict(color='black', dash='dash')),
    dict(type='line', x0=0.5, x1=0.5, y0=0, y1=10,
         line=dict(color='black', dash='dash'))
]

# Create figure
fig = go.Figure(data=traces + legend_traces)
fig.update_layout(
    title=config[first_comparison]['title'],
    xaxis=dict(title='log2(Fold Change)', gridcolor='gray'),
    yaxis=dict(title='-log10(p-value)', gridcolor='gray'),
    plot_bgcolor='lightslategray',
    paper_bgcolor='lightslategray',
    font=dict(color='black'),
    updatemenus=[dict(
        type='dropdown',
        x=1.15,
        y=1.15,
        showactive=True,
        buttons=buttons,
        direction='down',
        pad=dict(r=10, t=10)
    )],
    shapes=threshold_lines,
    showlegend=True
)

# Save output
fig.write_html('volcano_plot.html')