#!/usr/bin/env python3
"""
Trajectory Statistics and Visualization for Artsiv

This script analyzes conversation trajectories from Artsiv experiments and generates
beautiful visualizations for research presentations.

Usage:
    python trajectory_analyzer.py --input /path/to/output.jsonl --output /path/to/plots/
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
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
LOG = logging.getLogger(__name__)

# Set style for beautiful plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

class TrajectoryAnalyzer:
    """Analyzes Artsiv conversation trajectories and generates visualizations."""
    
    def __init__(self, input_file: str, output_dir: str):
        self.input_file = Path(input_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.data = []
        self.stats = {}
        
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
        
    def extract_trajectory_stats(self) -> Dict[str, Any]:
        """Extract comprehensive statistics from trajectories."""
        LOG.info("Extracting trajectory statistics...")
        
        # Initialize counters
        num_turns_list = []
        tool_usage = Counter()
        tool_sequences = []
        intervention_types = Counter()
        status_counts = Counter()
        retry_counts = []
        final_turn_injections = 0
        length_warnings = 0
        loop_interventions = 0
        context_truncations = 0
        
        # Token statistics
        total_tokens = []
        llm_tokens = []
        tool_tokens = []
        input_tokens = []
        
        # Success/failure analysis
        success_turn_counts = []
        failed_turn_counts = []
        
        for entry in self.data:
            # Basic stats
            status = entry.get('status', 'unknown')
            status_counts[status] += 1
            
            turns = entry.get('turns', [])
            num_turns = len(turns)
            num_turns_list.append(num_turns)
            
            if status == 'success':
                success_turn_counts.append(num_turns)
            elif status == 'failed':
                failed_turn_counts.append(num_turns)
            
            # Analyze each turn
            trajectory_tools = []
            trajectory_retries = 0
            trajectory_total_tokens = 0
            trajectory_llm_tokens = 0
            trajectory_tool_tokens = 0
            trajectory_input_tokens = 0
            
            for turn in turns:
                # Tool usage
                tool_call = turn.get('tool_call')
                if tool_call and isinstance(tool_call, dict):
                    tool_name = tool_call.get('tool', 'unknown')
                    tool_usage[tool_name] += 1
                    trajectory_tools.append(tool_name)
                
                # Intervention types
                turn_type = turn.get('turn_type')
                if turn_type:
                    intervention_types[turn_type] += 1
                    if turn_type == 'final_turn_instruction':
                        final_turn_injections += 1
                    elif turn_type == 'length_warning':
                        length_warnings += 1
                    elif turn_type == 'loop_intervention':
                        loop_interventions += 1
                
                # Retry counts
                retry_count = turn.get('_retry_count', 0)
                if retry_count > 0:
                    trajectory_retries += retry_count
                
                # Token counts
                llm_tokens_turn = turn.get('_llm_tokens', 0)
                tool_tokens_turn = turn.get('_tool_tokens', 0)
                input_tokens_turn = turn.get('_input_tokens', 0)
                
                trajectory_llm_tokens += llm_tokens_turn
                trajectory_tool_tokens += tool_tokens_turn
                trajectory_input_tokens += input_tokens_turn
                
                # Context truncation detection
                context_turn_ids = turn.get('_context_turn_ids', [])
                if context_turn_ids and len(context_turn_ids) < len(turns[:turns.index(turn)+1]):
                    context_truncations += 1
            
            # Store trajectory-level stats
            if trajectory_tools:
                tool_sequences.append(trajectory_tools)
            
            if trajectory_retries > 0:
                retry_counts.append(trajectory_retries)
            
            trajectory_total_tokens = trajectory_llm_tokens + trajectory_tool_tokens + trajectory_input_tokens
            if trajectory_total_tokens > 0:
                total_tokens.append(trajectory_total_tokens)
                llm_tokens.append(trajectory_llm_tokens)
                tool_tokens.append(trajectory_tool_tokens)
                input_tokens.append(trajectory_input_tokens)
        
        # Compile statistics
        self.stats = {
            'total_trajectories': len(self.data),
            'status_distribution': dict(status_counts),
            
            # Turn statistics
            'turns': {
                'mean': np.mean(num_turns_list) if num_turns_list else 0,
                'std': np.std(num_turns_list) if num_turns_list else 0,
                'min': min(num_turns_list) if num_turns_list else 0,
                'max': max(num_turns_list) if num_turns_list else 0,
                'median': np.median(num_turns_list) if num_turns_list else 0,
                'distribution': num_turns_list,
                'success_turns': success_turn_counts,
                'failed_turns': failed_turn_counts,
            },
            
            # Tool usage
            'tools': {
                'usage_counts': dict(tool_usage),
                'total_calls': sum(tool_usage.values()),
                'unique_tools': len(tool_usage),
                'sequences': tool_sequences,
            },
            
            # Intervention statistics
            'interventions': {
                'types': dict(intervention_types),
                'final_turn_injections': final_turn_injections,
                'length_warnings': length_warnings,
                'loop_interventions': loop_interventions,
                'context_truncations': context_truncations,
                'retry_counts': retry_counts,
                'total_retries': sum(retry_counts) if retry_counts else 0,
            },
            
            # Token statistics
            'tokens': {
                'total': {
                    'mean': np.mean(total_tokens) if total_tokens else 0,
                    'std': np.std(total_tokens) if total_tokens else 0,
                    'distribution': total_tokens,
                },
                'llm': {
                    'mean': np.mean(llm_tokens) if llm_tokens else 0,
                    'std': np.std(llm_tokens) if llm_tokens else 0,
                    'distribution': llm_tokens,
                },
                'tool': {
                    'mean': np.mean(tool_tokens) if tool_tokens else 0,
                    'std': np.std(tool_tokens) if tool_tokens else 0,
                    'distribution': tool_tokens,
                },
                'input': {
                    'mean': np.mean(input_tokens) if input_tokens else 0,
                    'std': np.std(input_tokens) if input_tokens else 0,
                    'distribution': input_tokens,
                },
            }
        }
        
        return self.stats
    
    def plot_turn_distribution(self):
        """Plot distribution of turns per trajectory."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Histogram with KDE
        turns_data = self.stats['turns']['distribution']
        
        ax1.hist(turns_data, bins=20, alpha=0.7, density=True, color='skyblue', edgecolor='black')
        
        # Add KDE if we have enough data
        if len(turns_data) > 1:
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(turns_data)
            x_range = np.linspace(min(turns_data), max(turns_data), 100)
            ax1.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
        
        ax1.axvline(self.stats['turns']['mean'], color='red', linestyle='--', 
                   label=f"Mean: {self.stats['turns']['mean']:.1f}")
        ax1.axvline(self.stats['turns']['median'], color='green', linestyle='--', 
                   label=f"Median: {self.stats['turns']['median']:.1f}")
        
        ax1.set_xlabel('Number of Turns', fontsize=12)
        ax1.set_ylabel('Density', fontsize=12)
        ax1.set_title('Distribution of Turns per Trajectory', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Box plot comparing success vs failed
        success_turns = self.stats['turns']['success_turns']
        failed_turns = self.stats['turns']['failed_turns']
        
        if success_turns and failed_turns:
            box_data = [success_turns, failed_turns]
            box_labels = ['Success', 'Failed']
            
            bp = ax2.boxplot(box_data, labels=box_labels, patch_artist=True, notch=True)
            colors = ['lightgreen', 'lightcoral']
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
        
        ax2.set_ylabel('Number of Turns', fontsize=12)
        ax2.set_title('Turns by Success Status', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'turn_distribution.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'turn_distribution.pdf', bbox_inches='tight')
        plt.close()
        
    def plot_tool_usage(self):
        """Plot tool usage statistics."""
        tool_counts = self.stats['tools']['usage_counts']
        
        if not tool_counts:
            LOG.warning("No tool usage data found")
            return
            
        # Sort tools by usage
        sorted_tools = sorted(tool_counts.items(), key=lambda x: x[1], reverse=True)
        tools, counts = zip(*sorted_tools)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
        
        # Bar chart of tool usage
        colors = plt.cm.Set3(np.linspace(0, 1, len(tools)))
        bars = ax1.bar(tools, counts, color=colors, edgecolor='black', alpha=0.8)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + max(counts)*0.01,
                    f'{int(height)}', ha='center', va='bottom', fontweight='bold')
        
        ax1.set_xlabel('Tool Name', fontsize=12)
        ax1.set_ylabel('Number of Calls', fontsize=12)
        ax1.set_title('Tool Usage Distribution', fontsize=14, fontweight='bold')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # Pie chart of tool usage proportions
        ax2.pie(counts, labels=tools, autopct='%1.1f%%', startangle=90, colors=colors)
        ax2.set_title('Tool Usage Proportions', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'tool_usage.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'tool_usage.pdf', bbox_inches='tight')
        plt.close()
        
    def plot_intervention_analysis(self):
        """Plot intervention and safety mechanism statistics."""
        interventions = self.stats['interventions']
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Intervention types
        if interventions['types']:
            int_types, int_counts = zip(*interventions['types'].items())
            colors = plt.cm.Pastel1(np.linspace(0, 1, len(int_types)))
            
            ax1.bar(int_types, int_counts, color=colors, edgecolor='black', alpha=0.8)
            ax1.set_xlabel('Intervention Type', fontsize=12)
            ax1.set_ylabel('Count', fontsize=12)
            ax1.set_title('Safety Intervention Types', fontsize=14, fontweight='bold')
            ax1.tick_params(axis='x', rotation=45)
            ax1.grid(True, alpha=0.3)
        
        # Retry distribution
        if interventions['retry_counts']:
            ax2.hist(interventions['retry_counts'], bins=10, alpha=0.7, 
                    color='orange', edgecolor='black')
            ax2.set_xlabel('Retries per Trajectory', fontsize=12)
            ax2.set_ylabel('Frequency', fontsize=12)
            ax2.set_title('Retry Distribution', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)
        
        # Summary statistics as text
        summary_text = f"""
        Intervention Summary:
        
        • Final Turn Injections: {interventions['final_turn_injections']}
        • Length Warnings: {interventions['length_warnings']}
        • Loop Interventions: {interventions['loop_interventions']}
        • Context Truncations: {interventions['context_truncations']}
        • Total Retries: {interventions['total_retries']}
        
        Safety Rate: {(sum(interventions['types'].values()) / self.stats['total_trajectories'] * 100):.1f}%
        """
        
        ax3.text(0.05, 0.95, summary_text, transform=ax3.transAxes, fontsize=11,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        ax3.set_xlim(0, 1)
        ax3.set_ylim(0, 1)
        ax3.axis('off')
        ax3.set_title('Intervention Summary', fontsize=14, fontweight='bold')
        
        # Status distribution
        status_dist = self.stats['status_distribution']
        if status_dist:
            status_labels, status_counts = zip(*status_dist.items())
            colors = ['lightgreen' if s == 'success' else 'lightcoral' if s == 'failed' else 'lightgray' 
                     for s in status_labels]
            
            ax4.pie(status_counts, labels=status_labels, autopct='%1.1f%%', 
                   startangle=90, colors=colors)
            ax4.set_title('Trajectory Status Distribution', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'intervention_analysis.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'intervention_analysis.pdf', bbox_inches='tight')
        plt.close()
        
    def plot_token_analysis(self):
        """Plot token usage analysis."""
        tokens = self.stats['tokens']
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Token distribution by type
        token_types = ['LLM', 'Tool', 'Input']
        token_means = [tokens['llm']['mean'], tokens['tool']['mean'], tokens['input']['mean']]
        token_stds = [tokens['llm']['std'], tokens['tool']['std'], tokens['input']['std']]
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        bars = ax1.bar(token_types, token_means, yerr=token_stds, capsize=5, 
                      color=colors, alpha=0.8, edgecolor='black')
        
        ax1.set_ylabel('Average Tokens', fontsize=12)
        ax1.set_title('Token Usage by Type', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, mean in zip(bars, token_means):
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(token_means)*0.01,
                    f'{int(mean)}', ha='center', va='bottom', fontweight='bold')
        
        # Total token distribution
        if tokens['total']['distribution']:
            ax2.hist(tokens['total']['distribution'], bins=30, alpha=0.7, 
                    color='purple', edgecolor='black')
            ax2.axvline(tokens['total']['mean'], color='red', linestyle='--', 
                       label=f"Mean: {tokens['total']['mean']:.0f}")
            ax2.set_xlabel('Total Tokens per Trajectory', fontsize=12)
            ax2.set_ylabel('Frequency', fontsize=12)
            ax2.set_title('Total Token Distribution', fontsize=14, fontweight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # Stacked bar for token composition
        if tokens['llm']['distribution'] and tokens['tool']['distribution']:
            # Sample some trajectories for visualization
            sample_indices = np.random.choice(len(tokens['llm']['distribution']), 
                                            min(20, len(tokens['llm']['distribution'])), replace=False)
            
            llm_sample = [tokens['llm']['distribution'][i] for i in sample_indices]
            tool_sample = [tokens['tool']['distribution'][i] for i in sample_indices]
            input_sample = [tokens['input']['distribution'][i] for i in sample_indices]
            
            x_pos = range(len(sample_indices))
            
            ax3.bar(x_pos, llm_sample, label='LLM Tokens', color=colors[0], alpha=0.8)
            ax3.bar(x_pos, tool_sample, bottom=llm_sample, label='Tool Tokens', 
                   color=colors[1], alpha=0.8)
            bottom_values = [l + t for l, t in zip(llm_sample, tool_sample)]
            ax3.bar(x_pos, input_sample, bottom=bottom_values, label='Input Tokens', 
                   color=colors[2], alpha=0.8)
            
            ax3.set_xlabel('Sample Trajectories', fontsize=12)
            ax3.set_ylabel('Tokens', fontsize=12)
            ax3.set_title('Token Composition (Sample)', fontsize=14, fontweight='bold')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # Token efficiency scatter plot
        if tokens['llm']['distribution'] and self.stats['turns']['distribution']:
            llm_tokens_list = tokens['llm']['distribution']
            turns_list = self.stats['turns']['distribution'][:len(llm_tokens_list)]
            
            ax4.scatter(turns_list, llm_tokens_list, alpha=0.6, color='green', s=50)
            
            # Add trend line
            if len(turns_list) > 1:
                z = np.polyfit(turns_list, llm_tokens_list, 1)
                p = np.poly1d(z)
                ax4.plot(turns_list, p(turns_list), "r--", alpha=0.8, linewidth=2)
            
            ax4.set_xlabel('Number of Turns', fontsize=12)
            ax4.set_ylabel('LLM Tokens', fontsize=12)
            ax4.set_title('Token Efficiency vs Turns', fontsize=14, fontweight='bold')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'token_analysis.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'token_analysis.pdf', bbox_inches='tight')
        plt.close()
        
    def generate_summary_report(self):
        """Generate a comprehensive summary report."""
        # Calculate success metrics safely
        success_avg = np.mean(self.stats['turns']['success_turns']) if self.stats['turns']['success_turns'] else None
        failed_avg = np.mean(self.stats['turns']['failed_turns']) if self.stats['turns']['failed_turns'] else None
        
        success_turns_str = f"{success_avg:.1f}" if success_avg is not None else "N/A"
        failed_turns_str = f"{failed_avg:.1f}" if failed_avg is not None else "N/A"
        
        report = f"""
# Artsiv Trajectory Analysis Report

## Dataset Overview
- **Total Trajectories**: {self.stats['total_trajectories']:,}
- **Status Distribution**: {self.stats['status_distribution']}

## Turn Statistics
- **Average Turns**: {self.stats['turns']['mean']:.2f} ± {self.stats['turns']['std']:.2f}
- **Median Turns**: {self.stats['turns']['median']:.1f}
- **Range**: {self.stats['turns']['min']} - {self.stats['turns']['max']} turns

## Tool Usage
- **Unique Tools**: {self.stats['tools']['unique_tools']}
- **Total Tool Calls**: {self.stats['tools']['total_calls']:,}
- **Most Used Tool**: {max(self.stats['tools']['usage_counts'], key=self.stats['tools']['usage_counts'].get) if self.stats['tools']['usage_counts'] else 'N/A'}

## Safety Interventions
- **Total Interventions**: {sum(self.stats['interventions']['types'].values()) if self.stats['interventions']['types'] else 0}
- **Final Turn Injections**: {self.stats['interventions']['final_turn_injections']}
- **Length Warnings**: {self.stats['interventions']['length_warnings']}
- **Loop Interventions**: {self.stats['interventions']['loop_interventions']}
- **Context Truncations**: {self.stats['interventions']['context_truncations']}
- **Total Retries**: {self.stats['interventions']['total_retries']}

## Token Usage
- **Average Total Tokens**: {self.stats['tokens']['total']['mean']:.0f} ± {self.stats['tokens']['total']['std']:.0f}
- **Average LLM Tokens**: {self.stats['tokens']['llm']['mean']:.0f} ± {self.stats['tokens']['llm']['std']:.0f}
- **Average Tool Tokens**: {self.stats['tokens']['tool']['mean']:.0f} ± {self.stats['tokens']['tool']['std']:.0f}
- **Average Input Tokens**: {self.stats['tokens']['input']['mean']:.0f} ± {self.stats['tokens']['input']['std']:.0f}

## Success Metrics
- **Success Rate**: {(self.stats['status_distribution'].get('success', 0) / self.stats['total_trajectories'] * 100):.1f}%
- **Average Turns (Success)**: {success_turns_str}
- **Average Turns (Failed)**: {failed_turns_str}

---
Generated by Artsiv Trajectory Analyzer
"""
        
        with open(self.output_dir / 'summary_report.md', 'w', encoding='utf-8') as f:
            f.write(report)
            
        LOG.info(f"Summary report saved to {self.output_dir / 'summary_report.md'}")
        
    def run_analysis(self):
        """Run complete trajectory analysis."""
        LOG.info("Starting trajectory analysis...")
        
        # Load and process data
        self.load_data()
        if not self.data:
            LOG.error("No data loaded. Exiting.")
            return
            
        # Extract statistics
        self.extract_trajectory_stats()
        
        # Generate visualizations
        LOG.info("Generating visualizations...")
        self.plot_turn_distribution()
        self.plot_tool_usage()
        self.plot_intervention_analysis()
        self.plot_token_analysis()
        
        # Generate summary report
        self.generate_summary_report()
        
        LOG.info(f"Analysis complete! Results saved to {self.output_dir}")
        LOG.info("Generated files:")
        for file in self.output_dir.glob("*"):
            LOG.info(f"  - {file.name}")


def main():
    parser = argparse.ArgumentParser(description="Analyze Artsiv trajectory data and generate visualizations")
    parser.add_argument('--input', '-i', required=True, help='Path to input JSONL file')
    parser.add_argument('--output', '-o', required=True, help='Output directory for plots and reports')
    
    args = parser.parse_args()
    
    analyzer = TrajectoryAnalyzer(args.input, args.output)
    analyzer.run_analysis()


if __name__ == "__main__":
    main()
