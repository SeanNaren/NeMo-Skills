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
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
LOG = logging.getLogger(__name__)

# Set professional style for plots
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

class DifficultyAnalyzer:
    """Analyzes trajectory characteristics across different difficulty levels."""
    
    def __init__(self, input_file: str, output_dir: str):
        self.input_file = Path(input_file)
        self.output_dir = Path(output_dir)
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
        
    def plot_difficulty_correlation(self):
        """Plot correlation between difficulty and various metrics."""
        if not self.difficulty_stats:
            LOG.warning("No difficulty statistics available")
            return
            
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Prepare data for plotting
        difficulties = list(self.difficulty_stats.keys())
        difficulty_order = ['<15 min fix', '15 min - 1 hour', '1-4 hours']  # Logical order
        difficulties = [d for d in difficulty_order if d in difficulties]
        
        colors = ['#2E86AB', '#A23B72', '#F18F01'][:len(difficulties)]
        
        # 1. Turn Distribution by Difficulty (Box Plot)
        turn_data = []
        labels = []
        for difficulty in difficulties:
            turn_data.append(self.difficulty_stats[difficulty]['turns']['distribution'])
            labels.append(difficulty)
        
        bp1 = ax1.boxplot(turn_data, labels=labels, patch_artist=True, notch=True)
        for patch, color in zip(bp1['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax1.set_ylabel('Number of Turns', fontweight='bold')
        ax1.set_title('Turn Distribution by Difficulty', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. Mean Turns vs Difficulty (Bar Plot)
        mean_turns = [self.difficulty_stats[d]['turns']['mean'] for d in difficulties]
        std_turns = [self.difficulty_stats[d]['turns']['std'] for d in difficulties]
        
        bars = ax2.bar(difficulties, mean_turns, yerr=std_turns, capsize=5, 
                      color=colors, alpha=0.8, edgecolor='black')
        
        # Add value labels on bars
        for bar, mean in zip(bars, mean_turns):
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(mean_turns)*0.02,
                    f'{mean:.1f}', ha='center', va='bottom', fontweight='bold')
        
        ax2.set_ylabel('Average Number of Turns', fontweight='bold')
        ax2.set_title('Average Turns by Difficulty', fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. Success Rate by Difficulty
        success_rates = [self.difficulty_stats[d]['success_rate'] for d in difficulties]
        
        bars = ax3.bar(difficulties, success_rates, color=colors, alpha=0.8, edgecolor='black')
        
        # Add value labels
        for bar, rate in zip(bars, success_rates):
            ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(success_rates)*0.01,
                    f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        ax3.set_ylabel('Success Rate (%)', fontweight='bold')
        ax3.set_title('Success Rate by Difficulty', fontweight='bold')
        ax3.set_ylim(0, 100)
        ax3.grid(True, alpha=0.3, axis='y')
        ax3.tick_params(axis='x', rotation=45)
        
        # 4. Statistical Summary Table
        ax4.axis('off')
        
        # Create summary statistics table
        table_data = []
        headers = ['Difficulty', 'Count', 'Avg Turns', 'Success Rate', 'Correlation']
        
        # Calculate correlation coefficient
        all_difficulties_encoded = []
        all_turns = []
        
        difficulty_encoding = {'<15 min fix': 1, '15 min - 1 hour': 2, '1-4 hours': 3}
        
        for difficulty in difficulties:
            entries = self.difficulty_stats[difficulty]['entries']
            for entry in entries:
                all_difficulties_encoded.append(difficulty_encoding[difficulty])
                all_turns.append(entry['num_turns'])
        
        correlation, p_value = stats.pearsonr(all_difficulties_encoded, all_turns) if len(all_turns) > 1 else (0, 1)
        
        for difficulty in difficulties:
            stats_data = self.difficulty_stats[difficulty]
            table_data.append([
                difficulty,
                f"{stats_data['count']}",
                f"{stats_data['turns']['mean']:.1f}",
                f"{stats_data['success_rate']:.1f}%",
                f"r={correlation:.3f}" if difficulty == difficulties[0] else ""
            ])
        
        # Add correlation p-value
        table_data.append(['', '', '', '', f"p={p_value:.3f}"])
        
        table = ax4.table(cellText=table_data, colLabels=headers, cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        # Style the table
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#4472C4')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        ax4.set_title('Statistical Summary', fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'difficulty_correlation_analysis.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'difficulty_correlation_analysis.pdf', bbox_inches='tight')
        plt.close()
        
    def plot_detailed_difficulty_analysis(self):
        """Create detailed analysis plots for difficulty correlation."""
        if not self.difficulty_stats:
            return
            
        # Create violin plots for detailed distribution analysis
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        difficulties = ['<15 min fix', '15 min - 1 hour', '1-4 hours']
        difficulties = [d for d in difficulties if d in self.difficulty_stats]
        
        # Prepare data for violin plots
        turn_data = []
        token_data = []
        labels = []
        
        for difficulty in difficulties:
            if difficulty in self.difficulty_stats:
                turn_data.append(self.difficulty_stats[difficulty]['turns']['distribution'])
                token_data.append(self.difficulty_stats[difficulty]['tokens']['distribution'])
                labels.append(difficulty)
        
        # Violin plot for turns
        parts1 = ax1.violinplot(turn_data, positions=range(len(labels)), showmeans=True, showmedians=True)
        ax1.set_xticks(range(len(labels)))
        ax1.set_xticklabels(labels, rotation=45)
        ax1.set_ylabel('Number of Turns', fontweight='bold')
        ax1.set_title('Turn Distribution by Difficulty (Detailed)', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Color the violin plots
        colors = ['#2E86AB', '#A23B72', '#F18F01']
        for pc, color in zip(parts1['bodies'], colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
        
        # Violin plot for tokens (if available)
        if any(token_data):
            parts2 = ax2.violinplot(token_data, positions=range(len(labels)), showmeans=True, showmedians=True)
            ax2.set_xticks(range(len(labels)))
            ax2.set_xticklabels(labels, rotation=45)
            ax2.set_ylabel('Total Tokens', fontweight='bold')
            ax2.set_title('Token Usage by Difficulty', fontweight='bold')
            ax2.grid(True, alpha=0.3)
            
            for pc, color in zip(parts2['bodies'], colors):
                pc.set_facecolor(color)
                pc.set_alpha(0.7)
        else:
            ax2.text(0.5, 0.5, 'Token data not available', ha='center', va='center', 
                    transform=ax2.transAxes, fontsize=14)
            ax2.set_title('Token Usage by Difficulty', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'difficulty_detailed_analysis.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'difficulty_detailed_analysis.pdf', bbox_inches='tight')
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
        
        # Generate visualizations
        LOG.info("Generating difficulty correlation visualizations...")
        self.plot_difficulty_correlation()
        self.plot_detailed_difficulty_analysis()
        
        # Generate report
        self.generate_difficulty_report()
        
        LOG.info(f"Difficulty analysis complete! Results saved to {self.output_dir}")
        LOG.info("Generated files:")
        for file in self.output_dir.glob("difficulty_*"):
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
