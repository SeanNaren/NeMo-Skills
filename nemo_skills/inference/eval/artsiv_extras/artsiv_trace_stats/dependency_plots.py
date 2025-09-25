#!/usr/bin/env python3
"""
Create clear dependency plots showing turns and tokens vs difficulty.
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
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18,
    'font.family': 'sans-serif',
})

def load_and_process_data(input_file):
    """Load and process the data for dependency analysis."""
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

def create_dependency_plots(difficulty_groups, output_dir):
    """Create clear dependency plots for turns and tokens vs difficulty."""
    output_dir = Path(output_dir)
    
    # Define difficulty order and colors
    difficulty_order = ['<15 min fix', '15 min - 1 hour', '1-4 hours']
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    
    # Create comprehensive figure
    fig = plt.figure(figsize=(20, 12))
    
    # Create grid layout
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # 1. Scatter plot: Turns vs Difficulty (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    
    all_x_turns = []
    all_y_turns = []
    
    for i, difficulty in enumerate(difficulty_order):
        if difficulty in difficulty_groups:
            entries = difficulty_groups[difficulty]
            turns_list = [e['turns'] for e in entries]
            x_positions = [i] * len(turns_list)
            
            # Add some jitter for better visualization
            x_jitter = np.random.normal(0, 0.1, len(x_positions))
            x_final = np.array(x_positions) + x_jitter
            
            ax1.scatter(x_final, turns_list, alpha=0.6, color=colors[i], s=30, label=difficulty)
            all_x_turns.extend([i] * len(turns_list))
            all_y_turns.extend(turns_list)
    
    # Add trend line
    if len(all_x_turns) > 1:
        z = np.polyfit(all_x_turns, all_y_turns, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(0, len(difficulty_order)-1, 100)
        ax1.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2, label=f'Trend (slope={z[0]:.2f})')
    
    ax1.set_xticks(range(len(difficulty_order)))
    ax1.set_xticklabels([d.replace(' - ', '-\n') for d in difficulty_order], rotation=0)
    ax1.set_ylabel('Number of Turns')
    ax1.set_title('Turns vs Difficulty Level', fontweight='bold')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # 2. Box plot: Turns vs Difficulty (top center)
    ax2 = fig.add_subplot(gs[0, 1])
    
    turns_data = []
    for difficulty in difficulty_order:
        if difficulty in difficulty_groups:
            entries = difficulty_groups[difficulty]
            turns_list = [e['turns'] for e in entries]
            turns_data.append(turns_list)
    
    bp = ax2.boxplot(turns_data, labels=[d.replace(' - ', '-\n') for d in difficulty_order], 
                     patch_artist=True, notch=True)
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax2.set_ylabel('Number of Turns')
    ax2.set_title('Turn Distribution by Difficulty', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 3. Mean and Std: Turns (top right)
    ax3 = fig.add_subplot(gs[0, 2])
    
    means_turns = []
    stds_turns = []
    counts = []
    
    for difficulty in difficulty_order:
        if difficulty in difficulty_groups:
            entries = difficulty_groups[difficulty]
            turns_list = [e['turns'] for e in entries]
            means_turns.append(np.mean(turns_list))
            stds_turns.append(np.std(turns_list))
            counts.append(len(turns_list))
    
    bars = ax3.bar(range(len(difficulty_order)), means_turns, yerr=stds_turns, 
                   capsize=5, color=colors, alpha=0.8, edgecolor='black')
    
    # Add value labels and sample sizes
    for i, (bar, mean, count) in enumerate(zip(bars, means_turns, counts)):
        ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(means_turns)*0.02,
                f'{mean:.2f}\n(n={count})', ha='center', va='bottom', fontweight='bold')
    
    ax3.set_xticks(range(len(difficulty_order)))
    ax3.set_xticklabels([d.replace(' - ', '-\n') for d in difficulty_order])
    ax3.set_ylabel('Average Number of Turns')
    ax3.set_title('Mean Turns by Difficulty', fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Statistics summary (top far right)
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.axis('off')
    
    # Calculate correlation
    correlation_turns, p_value_turns = stats.pearsonr(all_x_turns, all_y_turns) if len(all_x_turns) > 1 else (0, 1)
    
    stats_text = f"""
    TURNS vs DIFFICULTY
    
    Correlation: r = {correlation_turns:.3f}
    P-value: {p_value_turns:.3f}
    Significance: {'✓' if p_value_turns < 0.05 else '✗'} (α=0.05)
    
    Trend: {z[0]:.3f} turns/level
    
    Sample Sizes:
    """
    
    for difficulty in difficulty_order:
        if difficulty in difficulty_groups:
            count = len(difficulty_groups[difficulty])
            stats_text += f"\n{difficulty}: {count}"
    
    ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # 5-8. Same plots for TOKENS
    
    # 5. Scatter plot: Tokens vs Difficulty (middle left)
    ax5 = fig.add_subplot(gs[1, 0])
    
    all_x_tokens = []
    all_y_tokens = []
    
    for i, difficulty in enumerate(difficulty_order):
        if difficulty in difficulty_groups:
            entries = difficulty_groups[difficulty]
            tokens_list = [e['tokens'] for e in entries]
            x_positions = [i] * len(tokens_list)
            
            # Add some jitter
            x_jitter = np.random.normal(0, 0.1, len(x_positions))
            x_final = np.array(x_positions) + x_jitter
            
            ax5.scatter(x_final, tokens_list, alpha=0.6, color=colors[i], s=30, label=difficulty)
            all_x_tokens.extend([i] * len(tokens_list))
            all_y_tokens.extend(tokens_list)
    
    # Add trend line
    if len(all_x_tokens) > 1:
        z_tokens = np.polyfit(all_x_tokens, all_y_tokens, 1)
        p_tokens = np.poly1d(z_tokens)
        x_trend = np.linspace(0, len(difficulty_order)-1, 100)
        ax5.plot(x_trend, p_tokens(x_trend), "r--", alpha=0.8, linewidth=2, 
                label=f'Trend (slope={z_tokens[0]:.0f})')
    
    ax5.set_xticks(range(len(difficulty_order)))
    ax5.set_xticklabels([d.replace(' - ', '-\n') for d in difficulty_order], rotation=0)
    ax5.set_ylabel('Total Tokens')
    ax5.set_title('Tokens vs Difficulty Level', fontweight='bold')
    ax5.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax5.grid(True, alpha=0.3)
    
    # 6. Box plot: Tokens vs Difficulty (middle center)
    ax6 = fig.add_subplot(gs[1, 1])
    
    tokens_data = []
    for difficulty in difficulty_order:
        if difficulty in difficulty_groups:
            entries = difficulty_groups[difficulty]
            tokens_list = [e['tokens'] for e in entries]
            tokens_data.append(tokens_list)
    
    bp_tokens = ax6.boxplot(tokens_data, labels=[d.replace(' - ', '-\n') for d in difficulty_order], 
                           patch_artist=True, notch=True)
    
    for patch, color in zip(bp_tokens['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax6.set_ylabel('Total Tokens')
    ax6.set_title('Token Distribution by Difficulty', fontweight='bold')
    ax6.grid(True, alpha=0.3)
    
    # 7. Mean and Std: Tokens (middle right)
    ax7 = fig.add_subplot(gs[1, 2])
    
    means_tokens = []
    stds_tokens = []
    
    for difficulty in difficulty_order:
        if difficulty in difficulty_groups:
            entries = difficulty_groups[difficulty]
            tokens_list = [e['tokens'] for e in entries]
            means_tokens.append(np.mean(tokens_list))
            stds_tokens.append(np.std(tokens_list))
    
    bars = ax7.bar(range(len(difficulty_order)), means_tokens, yerr=stds_tokens, 
                   capsize=5, color=colors, alpha=0.8, edgecolor='black')
    
    # Add value labels
    for i, (bar, mean, count) in enumerate(zip(bars, means_tokens, counts)):
        ax7.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(means_tokens)*0.02,
                f'{mean:.0f}\n(n={count})', ha='center', va='bottom', fontweight='bold')
    
    ax7.set_xticks(range(len(difficulty_order)))
    ax7.set_xticklabels([d.replace(' - ', '-\n') for d in difficulty_order])
    ax7.set_ylabel('Average Total Tokens')
    ax7.set_title('Mean Tokens by Difficulty', fontweight='bold')
    ax7.grid(True, alpha=0.3, axis='y')
    
    # 8. Statistics summary for tokens (middle far right)
    ax8 = fig.add_subplot(gs[1, 3])
    ax8.axis('off')
    
    # Calculate correlation for tokens
    correlation_tokens, p_value_tokens = stats.pearsonr(all_x_tokens, all_y_tokens) if len(all_x_tokens) > 1 else (0, 1)
    
    stats_text_tokens = f"""
    TOKENS vs DIFFICULTY
    
    Correlation: r = {correlation_tokens:.3f}
    P-value: {p_value_tokens:.3f}
    Significance: {'✓' if p_value_tokens < 0.05 else '✗'} (α=0.05)
    
    Trend: {z_tokens[0]:.0f} tokens/level
    
    Token Ranges:
    """
    
    for difficulty in difficulty_order:
        if difficulty in difficulty_groups:
            entries = difficulty_groups[difficulty]
            tokens_list = [e['tokens'] for e in entries]
            min_tokens, max_tokens = min(tokens_list), max(tokens_list)
            stats_text_tokens += f"\n{difficulty}:\n  {min_tokens:,} - {max_tokens:,}"
    
    ax8.text(0.05, 0.95, stats_text_tokens, transform=ax8.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    # 9-10. Combined analysis (bottom row)
    
    # 9. Turns vs Tokens correlation (bottom left, spans 2 columns)
    ax9 = fig.add_subplot(gs[2, 0:2])
    
    for i, difficulty in enumerate(difficulty_order):
        if difficulty in difficulty_groups:
            entries = difficulty_groups[difficulty]
            turns_list = [e['turns'] for e in entries]
            tokens_list = [e['tokens'] for e in entries]
            
            ax9.scatter(turns_list, tokens_list, alpha=0.6, color=colors[i], 
                       s=50, label=difficulty, edgecolors='black', linewidth=0.5)
    
    # Add overall trend line
    all_turns_flat = []
    all_tokens_flat = []
    for difficulty in difficulty_order:
        if difficulty in difficulty_groups:
            entries = difficulty_groups[difficulty]
            all_turns_flat.extend([e['turns'] for e in entries])
            all_tokens_flat.extend([e['tokens'] for e in entries])
    
    if len(all_turns_flat) > 1:
        z_combined = np.polyfit(all_turns_flat, all_tokens_flat, 1)
        p_combined = np.poly1d(z_combined)
        x_range = np.linspace(min(all_turns_flat), max(all_turns_flat), 100)
        ax9.plot(x_range, p_combined(x_range), "r--", alpha=0.8, linewidth=2, 
                label=f'Overall trend (r={np.corrcoef(all_turns_flat, all_tokens_flat)[0,1]:.3f})')
    
    ax9.set_xlabel('Number of Turns')
    ax9.set_ylabel('Total Tokens')
    ax9.set_title('Turns vs Tokens Relationship', fontweight='bold')
    ax9.legend()
    ax9.grid(True, alpha=0.3)
    
    # 10. Summary statistics table (bottom right, spans 2 columns)
    ax10 = fig.add_subplot(gs[2, 2:4])
    ax10.axis('off')
    
    # Create summary table
    table_data = []
    headers = ['Difficulty', 'Count', 'Avg Turns', 'Avg Tokens', 'Turn Range', 'Token Range']
    
    for difficulty in difficulty_order:
        if difficulty in difficulty_groups:
            entries = difficulty_groups[difficulty]
            turns_list = [e['turns'] for e in entries]
            tokens_list = [e['tokens'] for e in entries]
            
            table_data.append([
                difficulty.replace(' - ', '-\n'),
                f"{len(entries)}",
                f"{np.mean(turns_list):.2f}",
                f"{np.mean(tokens_list):,.0f}",
                f"{min(turns_list)}-{max(turns_list)}",
                f"{min(tokens_list):,}-{max(tokens_list):,}"
            ])
    
    table = ax10.table(cellText=table_data, colLabels=headers, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 2)
    
    # Style the table
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color rows by difficulty
    for i, color in enumerate(colors):
        for j in range(len(headers)):
            table[(i+1, j)].set_facecolor(color)
            table[(i+1, j)].set_alpha(0.3)
    
    ax10.set_title('Summary Statistics', fontweight='bold', pad=20)
    
    # Overall title
    fig.suptitle('Artsiv: Dependency Analysis - Turns & Tokens vs Problem Difficulty', 
                 fontsize=20, fontweight='bold', y=0.98)
    
    plt.savefig(output_dir / 'comprehensive_dependency_analysis.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'comprehensive_dependency_analysis.pdf', bbox_inches='tight')
    plt.close()
    
    print(f"Comprehensive dependency analysis saved to {output_dir}")
    
    # Print summary statistics
    print("\n=== DEPENDENCY ANALYSIS SUMMARY ===")
    print(f"Turns vs Difficulty: r = {correlation_turns:.3f}, p = {p_value_turns:.3f}")
    print(f"Tokens vs Difficulty: r = {correlation_tokens:.3f}, p = {p_value_tokens:.3f}")
    print(f"Turns vs Tokens: r = {np.corrcoef(all_turns_flat, all_tokens_flat)[0,1]:.3f}")

def main():
    parser = argparse.ArgumentParser(description="Create dependency analysis plots")
    parser.add_argument('--input', '-i', required=True, help='Path to input JSONL file')
    parser.add_argument('--output', '-o', required=True, help='Output directory')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading and processing data...")
    difficulty_groups = load_and_process_data(args.input)
    
    print("Creating comprehensive dependency plots...")
    create_dependency_plots(difficulty_groups, output_dir)

if __name__ == "__main__":
    main()


