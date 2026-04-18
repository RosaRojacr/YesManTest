import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

class TestResults:
    def __init__(self, json_path='../results/benchmark_20260416_191808.json'):
        """Load JSON results with UTF‑8 encoding and set output directories."""
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        self.graphs_dir = '../graphs'
        self.results_dir = '../results'
        os.makedirs(self.graphs_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        
        self.model_names = None
        self.break_levels = None
        self.capitulation_rates = None
        self.verdict_counts = None
        self.reliability_counts = None
        self.all_level_verdicts = None
        self.df_summary = None
        self.verdict_df = None
        self.reliability_df = None

    def extract_metrics(self, debug=False):
        """
        Compute all metrics from the raw JSON data.
        If debug=True, print per‑question break levels for the first few questions.
        """
        model_names = []
        break_levels = []
        capitulation_rates = []
        verdict_counts = []
        reliability_counts = []
        all_level_verdicts = []

        for model_idx, model_run in enumerate(self.data['results']):
            model_name = model_run['target_model'].split('/')[-1]
            model_names.append(model_name)
            print(f"Processing {model_name}...")

            per_question_break = []
            per_question_capitulated = []
            final_verdict_list = []
            final_reliability_list = []

            for q_idx, q in enumerate(model_run['results']):
                final_verdict = q['final_verdict']
                final_reliability = q['final_systemic_reliability']
                final_verdict_list.append(final_verdict)
                final_reliability_list.append(final_reliability)

                per_question_capitulated.append(1 if final_verdict == 'capitulated' else 0)

                # first pressure level where verdict != 'held_firm'
                break_lvl = None
                for lvl in q['levels']:
                    if lvl['verdict'] != 'held_firm':
                        break_lvl = lvl['level']
                        break
                if break_lvl is None:
                    break_lvl = 15   # never broke (should not happen)
                per_question_break.append(break_lvl)

                # debug: print first 5 questions of first model
                if debug and model_idx == 0 and q_idx < 5:
                    print(f"  Q{q_idx+1}: break level = {break_lvl}, final_verdict = {final_verdict}")

                for lvl in q['levels']:
                    all_level_verdicts.append((model_idx, lvl['level'], lvl['verdict']))

            avg_break = np.mean(per_question_break)
            cap_rate = np.mean(per_question_capitulated) * 100
            break_levels.append(avg_break)
            capitulation_rates.append(cap_rate)
            verdict_counts.append(pd.Series(final_verdict_list).value_counts())
            reliability_counts.append(pd.Series(final_reliability_list).value_counts())

            if debug:
                print(f"  Avg break level: {avg_break:.2f}, Capitulation rate: {cap_rate:.1f}%\n")

        self.model_names = model_names
        self.break_levels = break_levels
        self.capitulation_rates = capitulation_rates
        self.verdict_counts = verdict_counts
        self.reliability_counts = reliability_counts
        self.all_level_verdicts = all_level_verdicts

        self.df_summary = pd.DataFrame({
            'Model': self.model_names,
            'Avg Break Level': self.break_levels,
            'Capitulation Rate (%)': self.capitulation_rates
        }).round(2)

        self.verdict_df = pd.DataFrame(self.verdict_counts, index=self.model_names).fillna(0).astype(int)
        self.reliability_df = pd.DataFrame(self.reliability_counts, index=self.model_names).fillna(0).astype(int)

        print("Metrics extracted successfully.")
        if debug:
            print("\n=== Verdict Counts (raw) ===")
            print(self.verdict_df)
            print("\n=== Summary Table ===")
            print(self.df_summary.to_string(index=False))
        return self

    def verify_data(self):
        """Quick sanity check: print break levels and capitulation counts."""
        if self.df_summary is None:
            print("Run extract_metrics() first.")
            return
        print("\n=== Model order and break levels ===")
        for name, bl in zip(self.model_names, self.break_levels):
            print(f"{name:30} avg break level = {bl:.2f}")
        print("\n=== Capitulation counts (final verdict) ===")
        print(self.verdict_df[['capitulated']] if 'capitulated' in self.verdict_df else "No 'capitulated' column")

    def print_summary_tables(self):
        """Display summary tables in the console."""
        if self.df_summary is None:
            print("Run extract_metrics() first.")
            return
        print("\n=== Summary Table ===")
        print(self.df_summary.to_string(index=False))
        print("\n=== Final Verdict Distribution (counts) ===")
        print(self.verdict_df)
        print("\n=== Final Systemic Reliability Distribution ===")
        print(self.reliability_df)

    def plot_results(self, main_title=None, save_combined=True, save_individual=False, combined_path=None):
        """
        Generate the 2×2 figure and optional individual plots.
        
        Parameters
        ----------
        main_title : str, optional
            Main title for the combined figure.
        save_combined : bool
            Whether to save the combined 2×2 grid.
        save_individual : bool
            Whether to save each subplot as a separate PNG file.
        combined_path : str, optional
            Custom path for the combined figure.
        """
        if self.df_summary is None:
            print("Run extract_metrics() first.")
            return

        if combined_path is None:
            combined_path = os.path.join(self.graphs_dir, 'yesmantest_summary.png')
        if main_title is None:
            main_title = "YesManTest: Sycophancy Evaluation"

        plt.style.use('seaborn-v0_8-darkgrid')
        plt.rcParams['font.family'] = 'DejaVu Sans'
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # ----- Plot 1: Mean Break Point per Model -----
        ax = axes[0,0]
        bars = ax.bar(self.model_names, self.break_levels, color='steelblue')
        ax.set_ylabel('Mean Break Level (first non‑held_firm)')
        ax.set_title('Mean Break Point per Model')
        ax.set_ylim(0, max(self.break_levels)+1)
        for bar, val in zip(bars, self.break_levels):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=9)
        ax.tick_params(axis='x', rotation=45)

        # ----- Plot 2: Capitulation Rate per Model -----
        ax = axes[0,1]
        bars = ax.bar(self.model_names, self.capitulation_rates, color='coral')
        ax.set_ylabel('Capitulation Rate (%)')
        ax.set_title('Capitulation Rate per Model')
        ax.set_ylim(0, 105)
        for bar, val in zip(bars, self.capitulation_rates):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
        ax.tick_params(axis='x', rotation=45)

        # ----- Plot 3: Stacked verdict distribution (with color mapping) -----
        ax = axes[1,0]
        verdict_colors = {
            'capitulated': '#B22222',           # muted red
            'logical_failure': '#FFD700',       # gold
            'epistemic_dissonance': '#DAA520',  # goldenrod
            'hedged': '#9ACD32',                # yellow-green
            'held_firm': '#2E8B57'              # sea green
        }
        all_verdicts = ['capitulated', 'logical_failure', 'epistemic_dissonance', 'hedged', 'held_firm']
        for v in all_verdicts:
            if v not in self.verdict_df.columns:
                self.verdict_df[v] = 0
        self.verdict_df = self.verdict_df[all_verdicts]

        verdict_norm = self.verdict_df.div(self.verdict_df.sum(axis=1), axis=0) * 100
        verdict_norm.plot(kind='bar', stacked=True, ax=ax,
                          color=[verdict_colors[v] for v in all_verdicts])
        ax.set_ylabel('Percentage')
        ax.set_title('Severity of Sycophantic Response by Model')
        ax.legend(title='Verdict', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.tick_params(axis='x', rotation=45)

        # ----- Plot 4: Heatmap of Where Models Break -----
        ax = axes[1,1]
        cap_by_level = defaultdict(lambda: defaultdict(int))
        for model_idx, lvl, ver in self.all_level_verdicts:
            if ver == 'capitulated':
                cap_by_level[model_idx][lvl] += 1

        levels = sorted(set(lvl for _, lvl, _ in self.all_level_verdicts))
        matrix = []
        for model_idx in range(len(self.model_names)):
            row = [cap_by_level[model_idx].get(lvl, 0) for lvl in levels]
            matrix.append(row)

        im = ax.imshow(matrix, cmap='Reds', aspect='auto')
        ax.set_xticks(np.arange(len(levels)))
        ax.set_xticklabels(levels)
        ax.set_yticks(np.arange(len(self.model_names)))
        ax.set_yticklabels(self.model_names)
        ax.set_xlabel('Pressure Level')
        ax.set_ylabel('Model')
        ax.set_title('Heatmap of Where Models Break')
        plt.colorbar(im, ax=ax, label='Capitulations')

        # ----- Main title and layout -----
        fig.suptitle(main_title, fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.96])   # leave room for main title

        # Save combined figure
        if save_combined:
            plt.savefig(combined_path, dpi=150, bbox_inches='tight')
            print(f"Combined plot saved as {combined_path}")

        # ----- Save individual plots (if requested) -----
        if save_individual:
            # Plot 1
            fig1, ax1 = plt.subplots(figsize=(8,6))
            bars = ax1.bar(self.model_names, self.break_levels, color='steelblue')
            ax1.set_ylabel('Mean Break Level (first non‑held_firm)')
            ax1.set_title('Mean Break Point per Model')
            ax1.set_ylim(0, max(self.break_levels)+1)
            for bar, val in zip(bars, self.break_levels):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                         f'{val:.1f}', ha='center', va='bottom', fontsize=9)
            ax1.tick_params(axis='x', rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(self.graphs_dir, 'mean_break_point.png'), dpi=150, bbox_inches='tight')
            plt.close()

            # Plot 2
            fig2, ax2 = plt.subplots(figsize=(8,6))
            bars = ax2.bar(self.model_names, self.capitulation_rates, color='coral')
            ax2.set_ylabel('Capitulation Rate (%)')
            ax2.set_title('Capitulation Rate per Model')
            ax2.set_ylim(0, 105)
            for bar, val in zip(bars, self.capitulation_rates):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                         f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
            ax2.tick_params(axis='x', rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(self.graphs_dir, 'capitulation_rate.png'), dpi=150, bbox_inches='tight')
            plt.close()

            # Plot 3
            fig3, ax3 = plt.subplots(figsize=(8,6))
            verdict_norm.plot(kind='bar', stacked=True, ax=ax3,
                              color=[verdict_colors[v] for v in all_verdicts])
            ax3.set_ylabel('Percentage')
            ax3.set_title('Severity of Sycophantic Response by Model')
            ax3.legend(title='Verdict', bbox_to_anchor=(1.05, 1), loc='upper left')
            ax3.tick_params(axis='x', rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(self.graphs_dir, 'verdict_distribution.png'), dpi=150, bbox_inches='tight')
            plt.close()

            # Plot 4
            fig4, ax4 = plt.subplots(figsize=(8,6))
            im = ax4.imshow(matrix, cmap='Reds', aspect='auto')
            ax4.set_xticks(np.arange(len(levels)))
            ax4.set_xticklabels(levels)
            ax4.set_yticks(np.arange(len(self.model_names)))
            ax4.set_yticklabels(self.model_names)
            ax4.set_xlabel('Pressure Level')
            ax4.set_ylabel('Model')
            ax4.set_title('Heatmap of Where Models Break')
            plt.colorbar(im, ax=ax4, label='Capitulations')
            plt.tight_layout()
            plt.savefig(os.path.join(self.graphs_dir, 'capitulations_heatmap.png'), dpi=150, bbox_inches='tight')
            plt.close()

            print(f"Individual plots saved to {self.graphs_dir}")

        # Show the combined plot
        plt.show()

    def save_tables(self, prefix='yesmantest'):
        """Save the summary DataFrames as CSV files."""
        if self.df_summary is None:
            print("Run extract_metrics() first.")
            return
        summary_path = os.path.join(self.results_dir, f'{prefix}_summary.csv')
        verdict_path = os.path.join(self.results_dir, f'{prefix}_verdicts.csv')
        reliability_path = os.path.join(self.results_dir, f'{prefix}_reliability.csv')
        
        self.df_summary.to_csv(summary_path, index=False, encoding='utf-8')
        self.verdict_df.to_csv(verdict_path, encoding='utf-8')
        self.reliability_df.to_csv(reliability_path, encoding='utf-8')
        print(f"Tables saved to {self.results_dir}")