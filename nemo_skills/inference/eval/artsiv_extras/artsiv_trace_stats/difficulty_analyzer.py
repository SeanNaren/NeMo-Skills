#!/usr/bin/env python3
"""
Difficulty-Based Trajectory Analysis for Artsiv

This script analyzes the correlation between problem difficulty and trajectory characteristics,
focusing on turn count distribution across difficulty levels.

Usage:
    python difficulty_analyzer.py --input /path/to/output.jsonl --output /path/to/plots/
"""

import json
import argparse
import logging
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Any, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
LOG = logging.getLogger(__name__)

# --- Robust Bold Font Setup (matching trajectory_analyzer) ---------------
import os
import matplotlib
import matplotlib.font_manager as fm
import logging

def force_bold_font():
    """
    Use a reliable font that works well with bold weights.
    Priority: Arial > Helvetica > system sans-serif
    """
    logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

    # Use reliable system fonts that have proper bold variants
    matplotlib.rcParams['font.family'] = ['Arial', 'Helvetica', 'DejaVu Sans', 'sans-serif']
    matplotlib.rcParams['font.weight'] = 'bold'
    matplotlib.rcParams['axes.unicode_minus'] = False
    
    # Force bold rendering with additional settings
    matplotlib.rcParams['axes.labelweight'] = 'bold'
    matplotlib.rcParams['axes.titleweight'] = 'bold'
    matplotlib.rcParams['figure.titleweight'] = 'bold'

    # Log what Matplotlib will actually draw with
    try:
        font_family = matplotlib.rcParams['font.family']
        # Handle case where font.family might be a list
        if isinstance(font_family, list):
            font_family = font_family[0] if font_family else 'Arial'
        
        final_path = fm.findfont(font_family, fontext='ttf', fallback_to_default=True)
        print(f"[Font] Using: {font_family}  "
              f"(weight={matplotlib.rcParams['font.weight']})")
        print(f"[Font] Resolved file: {final_path}")
    except Exception as e:
        print(f"[Font] Could not resolve final font path: {e}")

force_bold_font()
# -----------------------------------------------------------------------------

# Set style for beautiful plots (matching trajectory_analyzer)
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Global typography (all bold with reliable fonts - larger sizes, matching trajectory_analyzer)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans', 'sans-serif']
plt.rcParams['font.serif'] = ['Arial', 'Helvetica', 'DejaVu Sans', 'sans-serif']

def _bold_ticks_bold(ax):
    # Make tick labels bold with string weight
    for label in ax.get_xticklabels():
        label.set_fontweight('bold')
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')

class DifficultyAnalyzer:
    """Analyzes trajectory characteristics across different difficulty levels."""
    
    def __init__(self, input_file: str, output_dir: str):
        self.input_file = Path(input_file)
        # Add timestamp to output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(f"{output_dir}_{timestamp}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.data = []
        self.difficulty_stats = {}
        
    def load_data(self):
        """Load trajectory data from JSONL file."""
        LOG.info(f"Loading data from {self.input_file}")
        
        with open(self.input_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    entry = json.loads(line.strip())
                    if entry is not None:
                        self.data.append(entry)
                except json.JSONDecodeError as e:
                    LOG.warning(f"Skipping malformed JSON at line {line_num}: {e}")
                    
        LOG.info(f"Loaded {len(self.data)} trajectories")
        
    def extract_difficulty_stats(self) -> Dict[str, Any]:
        """Extract statistics grouped by difficulty level."""
        LOG.info("Extracting difficulty-based statistics...")
        
        # Group data by difficulty
        difficulty_groups = defaultdict(list)
        
        for entry in self.data:
            difficulty = entry.get('difficulty')
            if difficulty:  # Skip entries without difficulty field
                turns = entry.get('turns', [])
                num_turns = len(turns)
                status = entry.get('status', 'unknown')
                
                # Extract additional metrics
                total_tokens = 0
                tool_calls = 0
                
                for turn in turns:
                    # Token counts
                    llm_tokens = turn.get('_llm_tokens', 0)
                    tool_tokens = turn.get('_tool_tokens', 0)
                    input_tokens = turn.get('_input_tokens', 0)
                    total_tokens += llm_tokens + tool_tokens + input_tokens
                    
                    # Tool call counts
                    if turn.get('tool_call'):
                        tool_calls += 1
                
                difficulty_groups[difficulty].append({
                    'num_turns': num_turns,
                    'status': status,
                    'total_tokens': total_tokens,
                    'tool_calls': tool_calls,
                    'instance_id': entry.get('instance_id', 'unknown')
                })
        
        # Calculate statistics for each difficulty level
        stats = {}
        for difficulty, entries in difficulty_groups.items():
            if not entries:
                continue
                
            turns_list = [e['num_turns'] for e in entries]
            tokens_list = [e['total_tokens'] for e in entries]
            tool_calls_list = [e['tool_calls'] for e in entries]
            success_count = sum(1 for e in entries if e['status'] == 'success')
            
            stats[difficulty] = {
                'count': len(entries),
                'turns': {
                    'mean': np.mean(turns_list),
                    'std': np.std(turns_list),
                    'median': np.median(turns_list),
                    'min': min(turns_list),
                    'max': max(turns_list),
                    'distribution': turns_list
                },
                'tokens': {
                    'mean': np.mean(tokens_list) if tokens_list else 0,
                    'std': np.std(tokens_list) if tokens_list else 0,
                    'distribution': tokens_list
                },
                'tool_calls': {
                    'mean': np.mean(tool_calls_list) if tool_calls_list else 0,
                    'std': np.std(tool_calls_list) if tool_calls_list else 0,
                    'distribution': tool_calls_list
                },
                'success_rate': success_count / len(entries) * 100,
                'entries': entries
            }
        
        self.difficulty_stats = stats
        return stats
        

    def plot_individual_plots(self):
        """Create individual, focused plots for each analysis."""
        if not self.difficulty_stats:
            return
            
        LOG.info("Creating individual plots...")
        
        # Prepare data for individual plots
        difficulty_groups = {}
        for difficulty, stats_data in self.difficulty_stats.items():
            difficulty_groups[difficulty] = []
            for entry in stats_data['entries']:
                difficulty_groups[difficulty].append({
                    'turns': entry['num_turns'],
                    'tokens': entry['total_tokens']
                })
        
        # Create all individual plots
        self._plot_1_turns_scatter(difficulty_groups)
        self._plot_2_turns_boxplot(difficulty_groups)
        self._plot_3_tokens_scatter(difficulty_groups)
        self._plot_4_tokens_boxplot(difficulty_groups)
        self._plot_5_mean_comparison(difficulty_groups)
        self._plot_6_turns_vs_tokens(difficulty_groups)
        self._plot_7_violin_plots(difficulty_groups)
        self._plot_8_summary_statistics(difficulty_groups)
        
    def _plot_1_turns_scatter(self, difficulty_groups):
        """Plot 1: Scatter plot of Turns vs Difficulty."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        difficulty_order = ['<15 min fix', '15 min - 1 hour', '1-4 hours']
        colors = ['#F18F01', '#2E86AB', '#A23B72']
        
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
        ax.tick_params(axis='both', which='major', labelsize=12)
        _bold_ticks_bold(ax)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / '01_turns_vs_difficulty_scatter.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / '01_turns_vs_difficulty_scatter.pdf', bbox_inches='tight')
        plt.close()
        
    def _plot_2_turns_boxplot(self, difficulty_groups):
        """Plot 2: Box plot of Turns vs Difficulty."""
        fig, ax = plt.subplots(figsize=(8, 8))
        
        difficulty_order = ['<15 min fix', '15 min - 1 hour', '1-4 hours']
        colors = ['#F18F01', '#2E86AB', '#A23B72']
        
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
        ax.tick_params(axis='both', which='major', labelsize=12)
        _bold_ticks_bold(ax)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / '02_turns_vs_difficulty_boxplot.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / '02_turns_vs_difficulty_boxplot.pdf', bbox_inches='tight')
        plt.close()
        
    def _plot_3_tokens_scatter(self, difficulty_groups):
        """Plot 3: Scatter plot of Tokens vs Difficulty."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        difficulty_order = ['<15 min fix', '15 min - 1 hour', '1-4 hours']
        colors = ['#F18F01', '#2E86AB', '#A23B72']
        
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
        ax.tick_params(axis='both', which='major', labelsize=12)
        _bold_ticks_bold(ax)
        
        # Format y-axis with thousands separator
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / '03_tokens_vs_difficulty_scatter.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / '03_tokens_vs_difficulty_scatter.pdf', bbox_inches='tight')
        plt.close()
        
    def _plot_4_tokens_boxplot(self, difficulty_groups):
        """Plot 4: Box plot of Tokens vs Difficulty."""
        fig, ax = plt.subplots(figsize=(8, 8))
        
        difficulty_order = ['<15 min fix', '15 min - 1 hour', '1-4 hours']
        colors = ['#F18F01', '#2E86AB', '#A23B72']
        
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
        ax.tick_params(axis='both', which='major', labelsize=12)
        _bold_ticks_bold(ax)
        
        # Format y-axis with thousands separator
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / '04_tokens_vs_difficulty_boxplot.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / '04_tokens_vs_difficulty_boxplot.pdf', bbox_inches='tight')
        plt.close()
        
    def _plot_5_mean_comparison(self, difficulty_groups):
        """Plot 5: Separate bar charts for mean turns and mean tokens."""
        difficulty_order = ['<15 min fix', '15 min - 1 hour', '1-4 hours']
        colors = ['#F18F01', '#2E86AB', '#A23B72']
        
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
        
        # Plot 1: Mean Turns by Difficulty
        fig1, ax1 = plt.subplots(1, 1, figsize=(8, 6))
        bars1 = ax1.bar(range(len(difficulty_order)), means_turns, yerr=stds_turns, 
                        capsize=8, color=colors, alpha=0.8, edgecolor='black')
        
        # Add value labels aligned to the left
        for i, (bar, mean, count) in enumerate(zip(bars1, means_turns, counts)):
            ax1.text(bar.get_x(), bar.get_height() + max(means_turns)*0.01,
                    f'{mean:.2f}', ha='left', va='bottom', fontweight='bold', fontsize=15, color='black')
        
        ax1.set_xticks(range(len(difficulty_order)))
        ax1.set_xticklabels([d.replace(' - ', '-\n') for d in difficulty_order])
        ax1.set_ylabel('Average Number of Turns', fontweight='bold')
        ax1.set_title('Mean Turns by Difficulty', fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.set_ylim(bottom=0)
        ax1.tick_params(axis='both', which='major', labelsize=12)
        _bold_ticks_bold(ax1)
        
        # Add more space at the top for the labels and error bars
        max_with_error = max(means_turns) + max(stds_turns)
        ax1.set_ylim(0, max_with_error * 1.2)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / '05_mean_turns_by_difficulty.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / '05_mean_turns_by_difficulty.pdf', bbox_inches='tight')
        plt.close()
        
        # Plot 2: Mean Tokens by Difficulty
        fig2, ax2 = plt.subplots(1, 1, figsize=(8, 6))
        bars2 = ax2.bar(range(len(difficulty_order)), means_tokens, yerr=stds_tokens, 
                        capsize=8, color=colors, alpha=0.8, edgecolor='black')
        
        # Add value labels aligned to the left with comma formatting
        for i, (bar, mean, count) in enumerate(zip(bars2, means_tokens, counts)):
            ax2.text(bar.get_x(), bar.get_height() + max(means_tokens)*0.01,
                    f'{int(mean):,}', ha='left', va='bottom', fontweight='bold', fontsize=15, color='black')
        
        ax2.set_xticks(range(len(difficulty_order)))
        ax2.set_xticklabels([d.replace(' - ', '-\n') for d in difficulty_order])
        ax2.set_ylabel('Average Total Tokens', fontweight='bold')
        ax2.set_title('Mean Tokens by Difficulty', fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.set_ylim(bottom=0)
        ax2.tick_params(axis='both', which='major', labelsize=12)
        _bold_ticks_bold(ax2)
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
        
        # Add more space at the top for the labels and error bars
        max_with_error = max(means_tokens) + max(stds_tokens)
        ax2.set_ylim(0, max_with_error * 1.2)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / '05_mean_tokens_by_difficulty.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / '05_mean_tokens_by_difficulty.pdf', bbox_inches='tight')
        plt.close()
        
    def _plot_6_turns_vs_tokens(self, difficulty_groups):
        """Plot 6: Turns vs Tokens correlation."""
        fig, ax = plt.subplots(figsize=(12, 10))
        
        difficulty_order = ['<15 min fix', '15 min - 1 hour', '1-4 hours']
        colors = ['#F18F01', '#2E86AB', '#A23B72']
        
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
        ax.tick_params(axis='both', which='major', labelsize=12)
        _bold_ticks_bold(ax)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / '06_turns_vs_tokens_correlation.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / '06_turns_vs_tokens_correlation.pdf', bbox_inches='tight')
        plt.close()
        
    def _plot_7_violin_plots(self, difficulty_groups):
        """Plot 7: Separate violin plots for detailed distribution."""
        difficulty_order = ['<15 min fix', '15 min - 1 hour', '1-4 hours']
        colors = ['#F18F01', '#2E86AB', '#A23B72']
        
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
        
        # First plot: Violin plot for turns
        fig1, ax1 = plt.subplots(1, 1, figsize=(10, 6))
        parts1 = ax1.violinplot(turns_data, positions=range(len(labels)), 
                               showmeans=True, showmedians=True, showextrema=True)
        
        for pc, color in zip(parts1['bodies'], colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
            pc.set_edgecolor('black')  # Black edges
            pc.set_linewidth(2)
        
        # Style the mean and median lines in black with different line styles
        parts1['cmeans'].set_color('black')
        parts1['cmeans'].set_linewidth(3)
        parts1['cmeans'].set_linestyle('--')  # Dashed for mean
        parts1['cmedians'].set_color('black')
        parts1['cmedians'].set_linewidth(3)
        parts1['cmedians'].set_linestyle('-')  # Solid for median
        parts1['cmaxes'].set_color('black')
        parts1['cmaxes'].set_linewidth(2)
        parts1['cmins'].set_color('black')
        parts1['cmins'].set_linewidth(2)
        parts1['cbars'].set_color('black')
        parts1['cbars'].set_linewidth(2)
        
        ax1.set_xticks(range(len(labels)))
        ax1.set_xticklabels(labels)
        ax1.set_ylabel('Number of Turns', fontweight='bold')
        ax1.set_title('Turn Distribution (Detailed)', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(bottom=0)
        ax1.tick_params(axis='both', which='major', labelsize=12)
        _bold_ticks_bold(ax1)
        
        # Add legend for mean and median lines
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='black', linestyle='--', linewidth=3, label='Mean'),
            Line2D([0], [0], color='black', linestyle='-', linewidth=3, label='Median')
        ]
        ax1.legend(handles=legend_elements, loc='upper right', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / '07_turns_violin_distribution.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / '07_turns_violin_distribution.pdf', bbox_inches='tight')
        plt.close()
        
        # Second plot: Violin plot for tokens
        fig2, ax2 = plt.subplots(1, 1, figsize=(10, 6))
        parts2 = ax2.violinplot(tokens_data, positions=range(len(labels)), 
                               showmeans=True, showmedians=True, showextrema=True)
        
        for pc, color in zip(parts2['bodies'], colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
            pc.set_edgecolor('black')  # Black edges
            pc.set_linewidth(2)
        
        # Style the mean and median lines in black with different line styles
        parts2['cmeans'].set_color('black')
        parts2['cmeans'].set_linewidth(3)
        parts2['cmeans'].set_linestyle('--')  # Dashed for mean
        parts2['cmedians'].set_color('black')
        parts2['cmedians'].set_linewidth(3)
        parts2['cmedians'].set_linestyle('-')  # Solid for median
        parts2['cmaxes'].set_color('black')
        parts2['cmaxes'].set_linewidth(2)
        parts2['cmins'].set_color('black')
        parts2['cmins'].set_linewidth(2)
        parts2['cbars'].set_color('black')
        parts2['cbars'].set_linewidth(2)
        
        ax2.set_xticks(range(len(labels)))
        ax2.set_xticklabels(labels)
        ax2.set_ylabel('Total Tokens', fontweight='bold')
        ax2.set_title('Token Distribution (Detailed)', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(bottom=0)
        ax2.tick_params(axis='both', which='major', labelsize=12)
        _bold_ticks_bold(ax2)
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
        
        # Add legend for mean and median lines
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='black', linestyle='--', linewidth=3, label='Mean'),
            Line2D([0], [0], color='black', linestyle='-', linewidth=3, label='Median')
        ]
        ax2.legend(handles=legend_elements, loc='upper right', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / '07_tokens_violin_distribution.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / '07_tokens_violin_distribution.pdf', bbox_inches='tight')
        plt.close()
        
    def _plot_8_summary_statistics(self, difficulty_groups):
        """Plot 8: Summary statistics table."""
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.axis('off')
        
        difficulty_order = ['<15 min fix', '15 min - 1 hour', '1-4 hours']
        colors = ['#F18F01', '#2E86AB', '#A23B72']
        
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
        plt.savefig(self.output_dir / '08_summary_statistics_table.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / '08_summary_statistics_table.pdf', bbox_inches='tight')
        plt.close()

        
    def generate_difficulty_report(self):
        """Generate a comprehensive difficulty analysis report."""
        if not self.difficulty_stats:
            LOG.warning("No difficulty statistics to report")
            return
            
        # Calculate overall correlation
        all_difficulties_encoded = []
        all_turns = []
        
        difficulty_encoding = {'<15 min fix': 1, '15 min - 1 hour': 2, '1-4 hours': 3}
        
        for difficulty, stats_data in self.difficulty_stats.items():
            for entry in stats_data['entries']:
                all_difficulties_encoded.append(difficulty_encoding.get(difficulty, 0))
                all_turns.append(entry['num_turns'])
        
        correlation, p_value = stats.pearsonr(all_difficulties_encoded, all_turns) if len(all_turns) > 1 else (0, 1)
        
        report = f"""
# Artsiv Difficulty-Based Analysis Report

## Statistical Correlation
- **Pearson Correlation (Difficulty vs Turns)**: r = {correlation:.3f}
- **P-value**: {p_value:.3f}
- **Significance**: {'Significant' if p_value < 0.05 else 'Not significant'} (α = 0.05)

## Difficulty Level Analysis

"""
        
        # Add detailed stats for each difficulty level
        for difficulty in ['<15 min fix', '15 min - 1 hour', '1-4 hours']:
            if difficulty in self.difficulty_stats:
                stats_data = self.difficulty_stats[difficulty]
                report += f"""
### {difficulty}
- **Sample Size**: {stats_data['count']} trajectories
- **Average Turns**: {stats_data['turns']['mean']:.2f} ± {stats_data['turns']['std']:.2f}
- **Median Turns**: {stats_data['turns']['median']:.1f}
- **Range**: {stats_data['turns']['min']} - {stats_data['turns']['max']} turns
- **Success Rate**: {stats_data['success_rate']:.1f}%
- **Average Tokens**: {stats_data['tokens']['mean']:.0f} ± {stats_data['tokens']['std']:.0f}
"""
        
        report += f"""

## Key Findings

### Correlation Analysis
{'Strong' if abs(correlation) > 0.7 else 'Moderate' if abs(correlation) > 0.3 else 'Weak'} {'positive' if correlation > 0 else 'negative'} correlation between difficulty and turn count.

### Difficulty Trends
"""
        
        # Calculate trends
        difficulties = ['<15 min fix', '15 min - 1 hour', '1-4 hours']
        mean_turns = []
        success_rates = []
        
        for diff in difficulties:
            if diff in self.difficulty_stats:
                mean_turns.append(self.difficulty_stats[diff]['turns']['mean'])
                success_rates.append(self.difficulty_stats[diff]['success_rate'])
        
        if len(mean_turns) > 1:
            turn_trend = "increasing" if mean_turns[-1] > mean_turns[0] else "decreasing"
            success_trend = "decreasing" if success_rates[-1] < success_rates[0] else "stable"
            
            report += f"""
- **Turn Count Trend**: {turn_trend.capitalize()} with difficulty
- **Success Rate Trend**: {success_trend.capitalize()} across difficulty levels
- **Efficiency**: {'Higher' if correlation < 0 else 'Lower'} difficulty problems require {'fewer' if correlation < 0 else 'more'} turns on average
"""
        
        report += f"""

## Recommendations

Based on the correlation analysis:
1. {'Adjust timeout/step limits based on difficulty' if abs(correlation) > 0.3 else 'Current approach seems difficulty-agnostic'}
2. {'Consider difficulty-specific strategies' if abs(correlation) > 0.5 else 'Uniform strategy appears effective'}
3. Monitor performance on {max(self.difficulty_stats.keys(), key=lambda x: len(self.difficulty_stats[x]['entries']))} problems (largest category)

---
Generated by Artsiv Difficulty Analyzer
"""
        
        with open(self.output_dir / 'difficulty_analysis_report.md', 'w', encoding='utf-8') as f:
            f.write(report)
            
        LOG.info(f"Difficulty analysis report saved to {self.output_dir / 'difficulty_analysis_report.md'}")
        
    def run_analysis(self):
        """Run complete difficulty-based analysis."""
        LOG.info("Starting difficulty-based trajectory analysis...")
        
        # Load and process data
        self.load_data()
        if not self.data:
            LOG.error("No data loaded. Exiting.")
            return
            
        # Extract difficulty statistics
        self.extract_difficulty_stats()
        
        if not self.difficulty_stats:
            LOG.warning("No difficulty data found in the dataset. Skipping analysis.")
            return
        
        # Generate individual plots only
        LOG.info("Generating individual plots...")
        self.plot_individual_plots()
        
        # Generate report
        self.generate_difficulty_report()
        
        LOG.info(f"Difficulty analysis complete! Results saved to {self.output_dir}")
        LOG.info("Generated files:")
        for file in sorted(self.output_dir.glob("*")):
            LOG.info(f"  - {file.name}")


def main():
    parser = argparse.ArgumentParser(description="Analyze Artsiv trajectory difficulty correlation")
    parser.add_argument('--input', '-i', required=True, help='Path to input JSONL file')
    parser.add_argument('--output', '-o', required=True, help='Output directory for plots and reports')
    
    args = parser.parse_args()
    
    analyzer = DifficultyAnalyzer(args.input, args.output)
    analyzer.run_analysis()


if __name__ == "__main__":
    main()
