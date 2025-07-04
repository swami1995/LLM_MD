#!/usr/bin/env python3
"""
Script to extract and visualize projected_prices and projected_capital_shares from auditor analysis file
"""

import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def parse_mean_std(value_str):
    """Extract mean and std dev from string like 'Mean=2.6195, StdDev=1.2403'"""
    mean_match = re.search(r'Mean=([\d.-]+)', value_str)
    std_match = re.search(r'StdDev=([\d.-]+)', value_str)
    
    mean = float(mean_match.group(1)) if mean_match else 0.0
    std = float(std_match.group(1)) if std_match else 0.0
    
    return mean, std

def extract_and_plot_data(input_file, output_file):
    """
    Extract projected_prices and projected_capital_shares from the input file,
    save to output file, and create visualizations
    """
    
    with open(input_file, 'r') as f:
        content = f.read()
    
    lines = content.strip().split('\n')
    
    # Data structures to store parsed data
    data = {
        'projected_prices': {},
        'projected_capital_shares': {}
    }
    
    extracted_text = []
    current_agent = None
    current_dimension = None
    
    for line in lines:
        line = line.strip()
        
        # Parse agent/dimension header
        if line.startswith('Agent') and 'Dimension' in line:
            # Extract agent number and dimension name
            match = re.match(r'Agent (\d+), Dimension (.+):', line)
            if match:
                current_agent = int(match.group(1))
                current_dimension = match.group(2)
                extracted_text.append(f"{line}")
                
                # Initialize data structures
                if current_agent not in data['projected_prices']:
                    data['projected_prices'][current_agent] = {}
                    data['projected_capital_shares'][current_agent] = {}
        
        # Extract projected_prices
        elif 'projected_prices:' in line and current_agent is not None and current_dimension is not None:
            extracted_text.append(f"    - {line}")
            mean, std = parse_mean_std(line)
            data['projected_prices'][current_agent][current_dimension] = {'mean': mean, 'std': std}
        
        # Extract projected_capital_shares
        elif 'projected_capital_shares:' in line and current_agent is not None and current_dimension is not None:
            extracted_text.append(f"    - {line}")
            mean, std = parse_mean_std(line)
            data['projected_capital_shares'][current_agent][current_dimension] = {'mean': mean, 'std': std}
    
    # Save extracted text
    with open(output_file, 'w') as f:
        f.write('\n'.join(extracted_text))
    
    print(f"Extracted data saved to {output_file}")
    
    # Create visualizations
    create_visualizations(data)
    
    return data

def create_visualizations(data):
    """Create grouped bar chart visualizations of the data"""
    
    # Get all agents and dimensions
    agents = sorted(data['projected_prices'].keys())
    dimensions = sorted(set().union(*[agent_data.keys() for agent_data in data['projected_prices'].values()]))
    
    # Create DataFrame for easier plotting
    def create_dataframe(metric_name):
        rows = []
        for agent in agents:
            for dim in dimensions:
                if dim in data[metric_name][agent]:
                    rows.append({
                        'Agent': agent,
                        'Dimension': dim,
                        'Mean': data[metric_name][agent][dim]['mean'],
                        'StdDev': data[metric_name][agent][dim]['std']
                    })
        return pd.DataFrame(rows)
    
    df_prices = create_dataframe('projected_prices')
    df_shares = create_dataframe('projected_capital_shares')
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    
    # 1. Bar chart comparison by dimension for projected prices
    prices_grouped = df_prices.groupby(['Dimension', 'Agent']).first().reset_index()
    x_pos = np.arange(len(dimensions))
    width = 0.25
    
    for i, agent in enumerate(agents):
        agent_data = prices_grouped[prices_grouped['Agent'] == agent]
        agent_means = [agent_data[agent_data['Dimension'] == dim]['Mean'].iloc[0] if len(agent_data[agent_data['Dimension'] == dim]) > 0 else 0 for dim in dimensions]
        agent_stds = [agent_data[agent_data['Dimension'] == dim]['StdDev'].iloc[0] if len(agent_data[agent_data['Dimension'] == dim]) > 0 else 0 for dim in dimensions]
        
        ax1.bar(x_pos + i*width, agent_means, width, label=f'Agent {agent}', 
                yerr=agent_stds, capsize=3, alpha=0.8)
    
    ax1.set_xlabel('Dimensions')
    ax1.set_ylabel('Projected Prices (Mean)')
    ax1.set_title('Projected Prices by Dimension and Agent')
    ax1.set_xticks(x_pos + width)
    ax1.set_xticklabels(dimensions, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Bar chart comparison by dimension for projected capital shares
    shares_grouped = df_shares.groupby(['Dimension', 'Agent']).first().reset_index()
    
    for i, agent in enumerate(agents):
        agent_data = shares_grouped[shares_grouped['Agent'] == agent]
        agent_means = [agent_data[agent_data['Dimension'] == dim]['Mean'].iloc[0] if len(agent_data[agent_data['Dimension'] == dim]) > 0 else 0 for dim in dimensions]
        agent_stds = [agent_data[agent_data['Dimension'] == dim]['StdDev'].iloc[0] if len(agent_data[agent_data['Dimension'] == dim]) > 0 else 0 for dim in dimensions]
        
        ax2.bar(x_pos + i*width, agent_means, width, label=f'Agent {agent}', 
                yerr=agent_stds, capsize=3, alpha=0.8)
    
    ax2.set_xlabel('Dimensions')
    ax2.set_ylabel('Projected Capital Shares (Mean)')
    ax2.set_title('Projected Capital Shares by Dimension and Agent')
    ax2.set_xticks(x_pos + width)
    ax2.set_xticklabels(dimensions, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('outputs/debug/auditor_price_shares_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary statistics
    print("\n=== SUMMARY STATISTICS ===")
    print("\nProjected Prices:")
    print(df_prices.groupby('Agent')[['Mean', 'StdDev']].agg(['mean', 'std', 'min', 'max']).round(2))
    
    print("\nProjected Capital Shares:")
    print(df_shares.groupby('Agent')[['Mean', 'StdDev']].agg(['mean', 'std', 'min', 'max']).round(2))

if __name__ == "__main__":
    input_file = "outputs/debug/auditor_analysis.txt"
    output_file = "outputs/debug/extracted_auditor_price_shares.txt"
    
    data = extract_and_plot_data(input_file, output_file) 