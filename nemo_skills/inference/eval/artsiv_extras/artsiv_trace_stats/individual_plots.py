#!/usr/bin/env python3
"""
Create individual, focused plots for each analysis.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
from scipy import stats
import argparse

# Set professional style
plt.style.use('default')
sns.set_style("whitegrid")
plt.rcParams.update({
    'font.size': 16,
    'axes.titlesize': 20,
    'axes.labelsize': 18,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'figure.titlesize': 22,
    'font.family': 'sans-serif',
})

def load_data(input_file):
    """Load and process the data."""
    data = []
    with open(input_file, 'r') as f:
        for line in f:
            entry = json.loads(line.strip())
            if entry and entry.get('difficulty'):
                data.append(entry)
    
    # Group by difficulty
    difficulty_groups = defaultdict(list)
    for entry in data:
        difficulty = entry.get('difficulty')
        turns = entry.get('turns', [])
        num_turns = len(turns)
        
        # Calculate total tokens
        total_tokens = 0
        for turn in turns:
            llm_tokens = turn.get('_llm_tokens', 0)
            tool_tokens = turn.get('_tool_tokens', 0)
            input_tokens = turn.get('_input_tokens', 0)
            total_tokens += llm_tokens + tool_tokens + input_tokens
        
        difficulty_groups[difficulty].append({
            'turns': num_turns,
            'tokens': total_tokens
        })
    
    return difficulty_groups

def plot_1_turns_scatter(difficulty_groups, output_dir):
    """Plot 1: Scatter plot of Turns vs Difficulty."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    difficulty_order = ['<15 min fix', '15 min - 1 hour', '1-4 hours']
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    
    all_x = []
    all_y = []
    
    for i, difficulty in enumerate(difficulty_order):
        if difficulty in difficulty_groups:
            entries = difficulty_groups[difficulty]
            turns_list = [e['turns'] for e in entries]
            x_positions = [i] * len(turns_list)
            
            # Add jitter
            x_jitter = np.random.normal(0, 0.15, len(x_positions))
            x_final = np.array(x_positions) + x_jitter
            
            ax.scatter(x_final, turns_list, alpha=0.7, color=colors[i], s=60, 
                      label=f'{difficulty} (n={len(turns_list)})', edgecolors='black', linewidth=0.5)
            all_x.extend([i] * len(turns_list))
            all_y.extend(turns_list)
    
    # Add trend line
    if len(all_x) > 1:
        z = np.polyfit(all_x, all_y, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(-0.3, len(difficulty_order)-0.7, 100)
        ax.plot(x_trend, p(x_trend), "red", alpha=0.8, linewidth=3, 
                label=f'Trend: +{z[0]:.3f} turns/level')
        
        # Calculate and display correlation
        correlation, p_value = stats.pearsonr(all_x, all_y)
        ax.text(0.02, 0.98, f'r = {correlation:.3f}\np = {p_value:.3f}', 
                transform=ax.transAxes, fontsize=14, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_xticks(range(len(difficulty_order)))
    ax.set_xticklabels(difficulty_order, rotation=15, ha='right')
    ax.set_ylabel('Number of Turns', fontweight='bold')
    ax.set_xlabel('Problem Difficulty', fontweight='bold')
    ax.set_title('Number of Turns vs Problem Difficulty', fontweight='bold', pad=20)
    ax.legend(loc='upper left', bbox_to_anchor=(0.02, 0.85))
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig(output_dir / '01_turns_vs_difficulty_scatter.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / '01_turns_vs_difficulty_scatter.pdf', bbox_inches='tight')
    plt.close()

def plot_2_turns_boxplot(difficulty_groups, output_dir):
    """Plot 2: Box plot of Turns vs Difficulty."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    difficulty_order = ['<15 min fix', '15 min - 1 hour', '1-4 hours']
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    
    turns_data = []
    labels = []
    stats_text = []
    
    for difficulty in difficulty_order:
        if difficulty in difficulty_groups:
            entries = difficulty_groups[difficulty]
            turns_list = [e['turns'] for e in entries]
            turns_data.append(turns_list)
            labels.append(f'{difficulty}\n(n={len(turns_list)})')
            
            # Calculate stats for annotation
            mean_val = np.mean(turns_list)
            median_val = np.median(turns_list)
            stats_text.append(f'μ={mean_val:.2f}\nM={median_val:.1f}')
    
    bp = ax.boxplot(turns_data, tick_labels=labels, patch_artist=True, notch=True,
                    boxprops=dict(linewidth=2), whiskerprops=dict(linewidth=2),
                    capprops=dict(linewidth=2), medianprops=dict(linewidth=3, color='white'))
    
    # Color the boxes
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    
    # Add statistics annotations
    for i, (stats, color) in enumerate(zip(stats_text, colors)):
        ax.text(i+1, max([max(data) for data in turns_data]) * 0.95, stats,
                ha='center', va='top', fontweight='bold', fontsize=12,
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))
    
    ax.set_ylabel('Number of Turns', fontweight='bold')
    ax.set_xlabel('Problem Difficulty', fontweight='bold')
    ax.set_title('Turn Distribution by Problem Difficulty', fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig(output_dir / '02_turns_vs_difficulty_boxplot.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / '02_turns_vs_difficulty_boxplot.pdf', bbox_inches='tight')
    plt.close()

def plot_3_tokens_scatter(difficulty_groups, output_dir):
    """Plot 3: Scatter plot of Tokens vs Difficulty."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    difficulty_order = ['<15 min fix', '15 min - 1 hour', '1-4 hours']
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    
    all_x = []
    all_y = []
    
    for i, difficulty in enumerate(difficulty_order):
        if difficulty in difficulty_groups:
            entries = difficulty_groups[difficulty]
            tokens_list = [e['tokens'] for e in entries]
            x_positions = [i] * len(tokens_list)
            
            # Add jitter
            x_jitter = np.random.normal(0, 0.15, len(x_positions))
            x_final = np.array(x_positions) + x_jitter
            
            ax.scatter(x_final, tokens_list, alpha=0.7, color=colors[i], s=60, 
                      label=f'{difficulty} (n={len(tokens_list)})', edgecolors='black', linewidth=0.5)
            all_x.extend([i] * len(tokens_list))
            all_y.extend(tokens_list)
    
    # Add trend line
    if len(all_x) > 1:
        z = np.polyfit(all_x, all_y, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(-0.3, len(difficulty_order)-0.7, 100)
        ax.plot(x_trend, p(x_trend), "red", alpha=0.8, linewidth=3, 
                label=f'Trend: +{z[0]:,.0f} tokens/level')
        
        # Calculate and display correlation
        correlation, p_value = stats.pearsonr(all_x, all_y)
        ax.text(0.02, 0.98, f'r = {correlation:.3f}\np = {p_value:.3f}', 
                transform=ax.transAxes, fontsize=14, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_xticks(range(len(difficulty_order)))
    ax.set_xticklabels(difficulty_order, rotation=15, ha='right')
    ax.set_ylabel('Total Tokens', fontweight='bold')
    ax.set_xlabel('Problem Difficulty', fontweight='bold')
    ax.set_title('Token Usage vs Problem Difficulty', fontweight='bold', pad=20)
    ax.legend(loc='upper left', bbox_to_anchor=(0.02, 0.85))
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    
    # Format y-axis with thousands separator
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
    
    plt.tight_layout()
    plt.savefig(output_dir / '03_tokens_vs_difficulty_scatter.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / '03_tokens_vs_difficulty_scatter.pdf', bbox_inches='tight')
    plt.close()

def plot_4_tokens_boxplot(difficulty_groups, output_dir):
    """Plot 4: Box plot of Tokens vs Difficulty."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    difficulty_order = ['<15 min fix', '15 min - 1 hour', '1-4 hours']
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    
    tokens_data = []
    labels = []
    stats_text = []
    
    for difficulty in difficulty_order:
        if difficulty in difficulty_groups:
            entries = difficulty_groups[difficulty]
            tokens_list = [e['tokens'] for e in entries]
            tokens_data.append(tokens_list)
            labels.append(f'{difficulty}\n(n={len(tokens_list)})')
            
            # Calculate stats for annotation
            mean_val = np.mean(tokens_list)
            median_val = np.median(tokens_list)
            stats_text.append(f'μ={mean_val:,.0f}\nM={median_val:,.0f}')
    
    bp = ax.boxplot(tokens_data, tick_labels=labels, patch_artist=True, notch=True,
                    boxprops=dict(linewidth=2), whiskerprops=dict(linewidth=2),
                    capprops=dict(linewidth=2), medianprops=dict(linewidth=3, color='white'))
    
    # Color the boxes
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    
    # Add statistics annotations
    for i, (stats, color) in enumerate(zip(stats_text, colors)):
        ax.text(i+1, max([max(data) for data in tokens_data]) * 0.95, stats,
                ha='center', va='top', fontweight='bold', fontsize=12,
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))
    
    ax.set_ylabel('Total Tokens', fontweight='bold')
    ax.set_xlabel('Problem Difficulty', fontweight='bold')
    ax.set_title('Token Distribution by Problem Difficulty', fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    
    # Format y-axis with thousands separator
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
    
    plt.tight_layout()
    plt.savefig(output_dir / '04_tokens_vs_difficulty_boxplot.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / '04_tokens_vs_difficulty_boxplot.pdf', bbox_inches='tight')
    plt.close()

def plot_5_mean_comparison(difficulty_groups, output_dir):
    """Plot 5: Side-by-side bar chart comparing means."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    difficulty_order = ['<15 min fix', '15 min - 1 hour', '1-4 hours']
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    
    means_turns = []
    stds_turns = []
    means_tokens = []
    stds_tokens = []
    counts = []
    
    for difficulty in difficulty_order:
        if difficulty in difficulty_groups:
            entries = difficulty_groups[difficulty]
            turns_list = [e['turns'] for e in entries]
            tokens_list = [e['tokens'] for e in entries]
            
            means_turns.append(np.mean(turns_list))
            stds_turns.append(np.std(turns_list))
            means_tokens.append(np.mean(tokens_list))
            stds_tokens.append(np.std(tokens_list))
            counts.append(len(turns_list))
    
    # Plot 1: Turns
    bars1 = ax1.bar(range(len(difficulty_order)), means_turns, yerr=stds_turns, 
                    capsize=8, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    
    # Add value labels
    for i, (bar, mean, count) in enumerate(zip(bars1, means_turns, counts)):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(means_turns)*0.05,
                f'{mean:.2f}\n(n={count})', ha='center', va='bottom', fontweight='bold', fontsize=14)
    
    ax1.set_xticks(range(len(difficulty_order)))
    ax1.set_xticklabels([d.replace(' - ', '-\n') for d in difficulty_order])
    ax1.set_ylabel('Average Number of Turns', fontweight='bold')
    ax1.set_title('Mean Turns by Difficulty', fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(bottom=0)
    
    # Plot 2: Tokens
    bars2 = ax2.bar(range(len(difficulty_order)), means_tokens, yerr=stds_tokens, 
                    capsize=8, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    
    # Add value labels
    for i, (bar, mean, count) in enumerate(zip(bars2, means_tokens, counts)):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(means_tokens)*0.05,
                f'{mean:,.0f}\n(n={count})', ha='center', va='bottom', fontweight='bold', fontsize=14)
    
    ax2.set_xticks(range(len(difficulty_order)))
    ax2.set_xticklabels([d.replace(' - ', '-\n') for d in difficulty_order])
    ax2.set_ylabel('Average Total Tokens', fontweight='bold')
    ax2.set_title('Mean Tokens by Difficulty', fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(bottom=0)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
    
    plt.tight_layout()
    plt.savefig(output_dir / '05_mean_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / '05_mean_comparison.pdf', bbox_inches='tight')
    plt.close()

def plot_6_turns_vs_tokens(difficulty_groups, output_dir):
    """Plot 6: Turns vs Tokens correlation."""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    difficulty_order = ['<15 min fix', '15 min - 1 hour', '1-4 hours']
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    
    all_turns = []
    all_tokens = []
    
    for i, difficulty in enumerate(difficulty_order):
        if difficulty in difficulty_groups:
            entries = difficulty_groups[difficulty]
            turns_list = [e['turns'] for e in entries]
            tokens_list = [e['tokens'] for e in entries]
            
            ax.scatter(turns_list, tokens_list, alpha=0.7, color=colors[i], 
                      s=80, label=f'{difficulty} (n={len(turns_list)})', 
                      edgecolors='black', linewidth=0.5)
            
            all_turns.extend(turns_list)
            all_tokens.extend(tokens_list)
    
    # Add overall trend line
    if len(all_turns) > 1:
        z = np.polyfit(all_turns, all_tokens, 1)
        p = np.poly1d(z)
        x_range = np.linspace(min(all_turns), max(all_turns), 100)
        ax.plot(x_range, p(x_range), "red", alpha=0.8, linewidth=3, 
                label=f'Overall trend: r={np.corrcoef(all_turns, all_tokens)[0,1]:.3f}')
        
        # Add correlation info
        correlation = np.corrcoef(all_turns, all_tokens)[0,1]
        ax.text(0.02, 0.98, f'Pearson r = {correlation:.3f}\nSlope = {z[0]:,.0f} tokens/turn', 
                transform=ax.transAxes, fontsize=14, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    ax.set_xlabel('Number of Turns', fontweight='bold')
    ax.set_ylabel('Total Tokens', fontweight='bold')
    ax.set_title('Relationship: Turns vs Token Usage', fontweight='bold', pad=20)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
    
    plt.tight_layout()
    plt.savefig(output_dir / '06_turns_vs_tokens_correlation.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / '06_turns_vs_tokens_correlation.pdf', bbox_inches='tight')
    plt.close()

def plot_7_violin_plots(difficulty_groups, output_dir):
    """Plot 7: Violin plots for detailed distribution."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    difficulty_order = ['<15 min fix', '15 min - 1 hour', '1-4 hours']
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    
    # Prepare data
    turns_data = []
    tokens_data = []
    labels = []
    
    for difficulty in difficulty_order:
        if difficulty in difficulty_groups:
            entries = difficulty_groups[difficulty]
            turns_list = [e['turns'] for e in entries]
            tokens_list = [e['tokens'] for e in entries]
            turns_data.append(turns_list)
            tokens_data.append(tokens_list)
            labels.append(f'{difficulty}\n(n={len(turns_list)})')
    
    # Violin plot for turns
    parts1 = ax1.violinplot(turns_data, positions=range(len(labels)), 
                           showmeans=True, showmedians=True, showextrema=True)
    
    for pc, color in zip(parts1['bodies'], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)
        pc.set_edgecolor('black')
        pc.set_linewidth(1)
    
    ax1.set_xticks(range(len(labels)))
    ax1.set_xticklabels(labels)
    ax1.set_ylabel('Number of Turns', fontweight='bold')
    ax1.set_title('Turn Distribution (Detailed)', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)
    
    # Violin plot for tokens
    parts2 = ax2.violinplot(tokens_data, positions=range(len(labels)), 
                           showmeans=True, showmedians=True, showextrema=True)
    
    for pc, color in zip(parts2['bodies'], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)
        pc.set_edgecolor('black')
        pc.set_linewidth(1)
    
    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels(labels)
    ax2.set_ylabel('Total Tokens', fontweight='bold')
    ax2.set_title('Token Distribution (Detailed)', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(bottom=0)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
    
    plt.tight_layout()
    plt.savefig(output_dir / '07_violin_distributions.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / '07_violin_distributions.pdf', bbox_inches='tight')
    plt.close()

def plot_8_summary_statistics(difficulty_groups, output_dir):
    """Plot 8: Summary statistics table."""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('off')
    
    difficulty_order = ['<15 min fix', '15 min - 1 hour', '1-4 hours']
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    
    # Prepare table data
    table_data = []
    headers = ['Difficulty Level', 'Sample\nSize', 'Turns\nMean±SD', 'Turns\nMedian', 'Turns\nRange', 
               'Tokens\nMean±SD', 'Tokens\nMedian', 'Tokens\nRange']
    
    for difficulty in difficulty_order:
        if difficulty in difficulty_groups:
            entries = difficulty_groups[difficulty]
            turns_list = [e['turns'] for e in entries]
            tokens_list = [e['tokens'] for e in entries]
            
            table_data.append([
                difficulty,
                f"{len(entries)}",
                f"{np.mean(turns_list):.2f}±{np.std(turns_list):.2f}",
                f"{np.median(turns_list):.1f}",
                f"{min(turns_list)}-{max(turns_list)}",
                f"{np.mean(tokens_list):,.0f}±{np.std(tokens_list):,.0f}",
                f"{np.median(tokens_list):,.0f}",
                f"{min(tokens_list):,}-{max(tokens_list):,}"
            ])
    
    # Create table
    table = ax.table(cellText=table_data, colLabels=headers, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 3)
    
    # Style the table
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color rows by difficulty
    for i, color in enumerate(colors):
        for j in range(len(headers)):
            table[(i+1, j)].set_facecolor(color)
            table[(i+1, j)].set_alpha(0.3)
            table[(i+1, j)].set_text_props(weight='bold')
    
    ax.set_title('Comprehensive Statistical Summary', fontweight='bold', fontsize=24, pad=30)
    
    plt.tight_layout()
    plt.savefig(output_dir / '08_summary_statistics_table.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / '08_summary_statistics_table.pdf', bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Create individual dependency plots")
    parser.add_argument('--input', '-i', required=True, help='Path to input JSONL file')
    parser.add_argument('--output', '-o', required=True, help='Output directory')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading data...")
    difficulty_groups = load_data(args.input)
    
    print("Creating individual plots...")
    plot_1_turns_scatter(difficulty_groups, output_dir)
    print("  ✓ Plot 1: Turns vs Difficulty (Scatter)")
    
    plot_2_turns_boxplot(difficulty_groups, output_dir)
    print("  ✓ Plot 2: Turns vs Difficulty (Box Plot)")
    
    plot_3_tokens_scatter(difficulty_groups, output_dir)
    print("  ✓ Plot 3: Tokens vs Difficulty (Scatter)")
    
    plot_4_tokens_boxplot(difficulty_groups, output_dir)
    print("  ✓ Plot 4: Tokens vs Difficulty (Box Plot)")
    
    plot_5_mean_comparison(difficulty_groups, output_dir)
    print("  ✓ Plot 5: Mean Comparison (Bar Charts)")
    
    plot_6_turns_vs_tokens(difficulty_groups, output_dir)
    print("  ✓ Plot 6: Turns vs Tokens Correlation")
    
    plot_7_violin_plots(difficulty_groups, output_dir)
    print("  ✓ Plot 7: Violin Distribution Plots")
    
    plot_8_summary_statistics(difficulty_groups, output_dir)
    print("  ✓ Plot 8: Summary Statistics Table")
    
    print(f"\nAll individual plots saved to: {output_dir}")
    print("Generated files:")
    for file in sorted(output_dir.glob("*.png")):
        print(f"  - {file.name}")

if __name__ == "__main__":
    main()


