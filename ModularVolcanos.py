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
        
        # Check if the column is already numeric
        if pd.api.types.is_numeric_dtype(df[column_name]):
            return df[column_name]
        
        # If it's not numeric, then proceed with string cleaning
        if pd.api.types.is_string_dtype(df[column_name]):
            df[column_name] = df[column_name].str.strip()
        
        # Convert to numeric
        numeric_values = pd.to_numeric(df[column_name], errors='coerce')
        
        # Calculate the percentage of invalid (NaN) values
        invalid_percentage = numeric_values.isna().mean() * 100
        if invalid_percentage > 30:
            problematic_entries = df[df[column_name].isna()]
            problematic_sample = problematic_entries.sample(
                min(10, len(problematic_entries))
            ).index
            
            print(f"\nProblematic entries in '{column_name}':")
            for idx in problematic_sample:
                original_value = df.loc[idx, column_name]
                print(f"Index {idx}: '{original_value}'")
                
            raise ValueError(
                f"More than 30% ({invalid_percentage:.2f}%) of values in column '{column_name}' are invalid."
            )
        print(f"Column '{column_name}' cleaned successfully with {invalid_percentage:.2f}% invalid values.")
        
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
    # Additional comparisons omitted for brevity...
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
traces = []
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

    # Assign colors
    colors = []
    for idx in data.index:
        name = data.loc[idx, 'Name']
        fc = log2_ratio[idx]
        pval = neg_log_p[idx]

        if name in significant_compounds:
            colors.append('gold')
        elif not pd.isna(fc) and not pd.isna(pval):
            if fc > 0.5 and pval > -np.log10(0.05):
                colors.append('green')
            elif fc < -0.5 and pval > -np.log10(0.05):
                colors.append('red')
            else:
                colors.append('blue')
        else:
            colors.append('blue')

    traces.append(go.Scatter(
        x=log2_ratio,
        y=neg_log_p,
        mode='markers',
        marker=dict(color=colors, opacity=0.9),
        hovertext=hover_text,
        hoverinfo='text',
        visible=(comp == first_comparison),
        showlegend=False,
        hoverlabel=dict(
            bgcolor='red',
            bordercolor='black',
            font=dict(size=16, family='Arial')
        )
    ))

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