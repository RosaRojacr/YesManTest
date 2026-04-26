"""
plot_results.py - Analysis and plotting for sycophancy benchmark results.
"""

import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def list_model_names_in_file(json_path):
    """Print all target_model names found in a benchmark JSON file."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Models in {os.path.basename(json_path)}:")
    for entry in data['results']:
        print(f"  - {entry['target_model']}")

class TestResults:
    def __init__(self, json_path='../results/benchmark_20260416_191808.json'):
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
            total_questions = len(model_run['results'])
            skipped_count = 0

            for q_idx, q in enumerate(model_run['results']):
                if q.get('skipped') and q.get('skip_reason') == 'baseline_incorrect':
                    skipped_count += 1
                    continue

                final_verdict = q['final_verdict']
                final_reliability = q['final_systemic_reliability']
                final_verdict_list.append(final_verdict)
                final_reliability_list.append(final_reliability)
                per_question_capitulated.append(1 if final_verdict == 'capitulated' else 0)

                break_lvl = None
                for lvl in q['levels']:
                    if lvl['verdict'] != 'held_firm':
                        break_lvl = lvl['level']
                        break
                if break_lvl is None:
                    break_lvl = 15
                per_question_break.append(break_lvl)

                if debug and model_idx == 0 and len(per_question_break) <= 5:
                    print(f"  Q{q_idx+1}: break level = {break_lvl}, final_verdict = {final_verdict}")

                for lvl in q['levels']:
                    all_level_verdicts.append((model_idx, lvl['level'], lvl['verdict']))

            if skipped_count > 0:
                print(f"  Skipped {skipped_count}/{total_questions} questions (baseline incorrect)")

            if per_question_break:
                avg_break = np.mean(per_question_break)
                cap_rate = np.mean(per_question_capitulated) * 100
            else:
                avg_break = float('nan')
                cap_rate = float('nan')
                print(f"  WARNING: No valid questions for {model_name} after filtering.")

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
        if self.df_summary is None:
            print("Run extract_metrics() first.")
            return
        print("\n=== Model order and break levels ===")
        for name, bl in zip(self.model_names, self.break_levels):
            print(f"{name:30} avg break level = {bl:.2f}")
        print("\n=== Capitulation counts (final verdict) ===")
        print(self.verdict_df[['capitulated']] if 'capitulated' in self.verdict_df else "No 'capitulated' column")

    def print_summary_tables(self):
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
        if self.df_summary is None:
            print("Run extract_metrics() first.")
            return

        if combined_path is None:
            combined_path = os.path.join(self.graphs_dir, 'yesmantest_summary.png')
        if main_title is None:
            main_title = "YesManTest: Sycophancy Evaluation"

        # ------------------------------------------------------------------
        # Dynamic figure width to prevent label overlap
        # Base width per model, scaled by number of models and max label length
        # ------------------------------------------------------------------
        n_models = len(self.model_names)
        max_label_len = max(len(name) for name in self.model_names)
        # Heuristic: 1.2 inches per model, plus extra for longer names
        width = max(10, n_models * 1.4 + max_label_len * 0.12)
        fig, axes = plt.subplots(2, 2, figsize=(width, 10))

        plt.style.use('seaborn-v0_8-darkgrid')
        plt.rcParams['font.family'] = 'DejaVu Sans'

        # ----- Plot 1: Mean Break Point -----
        ax = axes[0,0]
        bars = ax.bar(self.model_names, self.break_levels, color='steelblue', width=0.7)
        ax.set_ylabel('Mean Break Level (first non‑held_firm)')
        ax.set_title('Mean Break Point per Model')
        max_val = max(self.break_levels)
        ax.set_ylim(0, max_val + 1 if not np.isnan(max_val) else 1)
        for bar, val in zip(bars, self.break_levels):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        f'{val:.1f}', ha='center', va='bottom', fontsize=9)
        # Labels horizontal, small font if many models
        plt.setp(ax.get_xticklabels(), rotation=0, ha='center', fontsize=min(10, 120/n_models))

        # ----- Plot 2: Capitulation Rate -----
        ax = axes[0,1]
        bars = ax.bar(self.model_names, self.capitulation_rates, color='coral', width=0.7)
        ax.set_ylabel('Capitulation Rate (%)')
        ax.set_title('Capitulation Rate per Model')
        ax.set_ylim(0, 105)
        for bar, val in zip(bars, self.capitulation_rates):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
        plt.setp(ax.get_xticklabels(), rotation=0, ha='center', fontsize=min(10, 120/n_models))

        # ----- Plot 3: Stacked verdicts -----
        ax = axes[1,0]
        verdict_colors = {
            'capitulated': '#B22222',
            'logical_failure': '#FFD700',
            'epistemic_dissonance': '#DAA520',
            'hedged': '#9ACD32',
            'held_firm': '#2E8B57'
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
        plt.setp(ax.get_xticklabels(), rotation=0, ha='center', fontsize=min(10, 120/n_models))

        # ----- Plot 4: Heatmap -----
        ax = axes[1,1]
        cap_by_level = defaultdict(lambda: defaultdict(int))
        for model_idx, lvl, ver in self.all_level_verdicts:
            if ver == 'capitulated':
                cap_by_level[model_idx][lvl] += 1

        levels = sorted(set(lvl for _, lvl, _ in self.all_level_verdicts))
        matrix = []
        for model_idx in range(n_models):
            row = [cap_by_level[model_idx].get(lvl, 0) for lvl in levels]
            matrix.append(row)

        im = ax.imshow(matrix, cmap='Reds', aspect='auto')
        ax.set_xticks(np.arange(len(levels)))
        ax.set_xticklabels(levels)
        ax.set_yticks(np.arange(n_models))
        ax.set_yticklabels(self.model_names, fontsize=min(10, 120/n_models))
        ax.set_xlabel('Pressure Level')
        ax.set_ylabel('Model')
        ax.set_title('Heatmap of Where Models Break')
        plt.colorbar(im, ax=ax, label='Capitulations')

        fig.suptitle(main_title, fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        if save_combined:
            plt.savefig(combined_path, dpi=150, bbox_inches='tight')
            print(f"Combined plot saved as {combined_path}")

        if save_individual:
            # individual plots also use the dynamic width scaling
            for i, (title, data, color, ylabel, suffix, is_percent) in enumerate([
                ("Mean Break Point per Model", self.break_levels, 'steelblue',
                 'Mean Break Level', 'mean_break_point', False),
                ("Capitulation Rate per Model", self.capitulation_rates, 'coral',
                 'Capitulation Rate (%)', 'capitulation_rate', True),
            ]):
                fig_i, ax_i = plt.subplots(figsize=(width*0.7, 6))
                bars = ax_i.bar(self.model_names, data, color=color, width=0.7)
                ax_i.set_ylabel(ylabel)
                ax_i.set_title(title)
                if is_percent:
                    ax_i.set_ylim(0, 105)
                else:
                    ax_i.set_ylim(0, max(data) + 1)
                for bar, val in zip(bars, data):
                    if not np.isnan(val):
                        offset = 0.1 if not is_percent else 1
                        ax_i.text(bar.get_x() + bar.get_width()/2, bar.get_height() + offset,
                                  f'{val:.1f}' + ('%' if is_percent else ''), ha='center', va='bottom', fontsize=9)
                plt.setp(ax_i.get_xticklabels(), rotation=0, ha='center', fontsize=min(10, 120/n_models))
                plt.tight_layout()
                plt.savefig(os.path.join(self.graphs_dir, f'{suffix}.png'), dpi=150, bbox_inches='tight')
                plt.close()

            # Verdict distribution
            fig3, ax3 = plt.subplots(figsize=(width*0.7, 6))
            verdict_norm.plot(kind='bar', stacked=True, ax=ax3,
                              color=[verdict_colors[v] for v in all_verdicts])
            ax3.set_ylabel('Percentage')
            ax3.set_title('Severity of Sycophantic Response by Model')
            ax3.legend(title='Verdict', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.setp(ax3.get_xticklabels(), rotation=0, ha='center', fontsize=min(10, 120/n_models))
            plt.tight_layout()
            plt.savefig(os.path.join(self.graphs_dir, 'verdict_distribution.png'), dpi=150, bbox_inches='tight')
            plt.close()

            # Heatmap
            fig4, ax4 = plt.subplots(figsize=(width*0.7, 6))
            im = ax4.imshow(matrix, cmap='Reds', aspect='auto')
            ax4.set_xticks(np.arange(len(levels)))
            ax4.set_xticklabels(levels)
            ax4.set_yticks(np.arange(n_models))
            ax4.set_yticklabels(self.model_names, fontsize=min(10, 120/n_models))
            ax4.set_xlabel('Pressure Level')
            ax4.set_ylabel('Model')
            ax4.set_title('Heatmap of Where Models Break')
            plt.colorbar(im, ax=ax4, label='Capitulations')
            plt.tight_layout()
            plt.savefig(os.path.join(self.graphs_dir, 'capitulations_heatmap.png'), dpi=150, bbox_inches='tight')
            plt.close()

            print(f"Individual plots saved to {self.graphs_dir}")

        plt.show()

    def save_tables(self, prefix='yesmantest'):
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

# =============================================================================
def compare_single_model_across_conditions(
    model_name_baseline,
    model_name_hardened,
    model_name_lora,
    model_name_lora_dpo,
    baseline_json_path,
    hardened_json_path,
    lora_json_path,
    lora_dpo_json_path,
    condition_labels=("Baseline", "Hardened Prompt", "LoRA‑Only", "LoRA + DPO"),
    output_dir="../graphs",
    show_plot=True,
    Title=("Title")
):
    import json, os, numpy as np, pandas as pd, matplotlib.pyplot as plt
    from collections import defaultdict

    def load_json(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    baseline_data = load_json(baseline_json_path)
    hardened_data = load_json(hardened_json_path)
    lora_data = load_json(lora_json_path)
    lora_dpo_data = load_json(lora_dpo_json_path)

    def extract_model_metrics(data, target_name):
        for entry in data['results']:
            if entry['target_model'] == target_name:
                results = entry['results']
                break
        else:
            raise ValueError(f"Model '{target_name}' not found in file.")

        per_question_break = []
        per_question_capitulated = []
        final_verdict_list = []
        all_level_verdicts = []
        skipped_count = 0
        total_questions = len(results)

        for q in results:
            if q.get('skipped') and q.get('skip_reason') == 'baseline_incorrect':
                skipped_count += 1
                continue

            final_verdict = q['final_verdict']
            final_verdict_list.append(final_verdict)
            per_question_capitulated.append(1 if final_verdict == 'capitulated' else 0)

            break_lvl = None
            for lvl in q['levels']:
                if lvl['verdict'] != 'held_firm':
                    break_lvl = lvl['level']
                    break
            if break_lvl is None:
                break_lvl = 15
            per_question_break.append(break_lvl)

            for lvl in q['levels']:
                all_level_verdicts.append((lvl['level'], lvl['verdict']))

        if skipped_count > 0:
            print(f"  Skipped {skipped_count}/{total_questions} questions (baseline incorrect) for {target_name}")

        if per_question_break:
            avg_break = np.mean(per_question_break)
            cap_rate = np.mean(per_question_capitulated) * 100
        else:
            avg_break = float('nan')
            cap_rate = float('nan')
            print(f"  WARNING: No valid questions for {target_name} after filtering.")

        verdict_counts = pd.Series(final_verdict_list).value_counts()
        return {
            'avg_break': avg_break,
            'cap_rate': cap_rate,
            'verdict_counts': verdict_counts,
            'all_level_verdicts': all_level_verdicts,
        }

    metrics_baseline = extract_model_metrics(baseline_data, model_name_baseline)
    metrics_hardened = extract_model_metrics(hardened_data, model_name_hardened)
    metrics_lora = extract_model_metrics(lora_data, model_name_lora)
    metrics_lora_dpo = extract_model_metrics(lora_dpo_data, model_name_lora_dpo)

    all_metrics = [metrics_baseline, metrics_hardened, metrics_lora, metrics_lora_dpo]
    labels = condition_labels

    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Subplot 1: Mean Break Point
    ax = axes[0, 0]
    break_vals = [m['avg_break'] for m in all_metrics]
    colors = ['steelblue', 'darkorange', 'mediumpurple', 'seagreen']
    bars = ax.bar(labels, break_vals, color=colors)
    ax.set_ylabel('Mean Break Level')
    ax.set_title('Mean Break Point by Condition')
    max_break = max([v for v in break_vals if not np.isnan(v)], default=0)
    ax.set_ylim(0, max_break + 1)
    ax.tick_params(axis='x', rotation=15)
    for bar, val in zip(bars, break_vals):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=9)

    # Subplot 2: Capitulation Rate
    ax = axes[0, 1]
    cap_vals = [m['cap_rate'] for m in all_metrics]
    bars = ax.bar(labels, cap_vals, color=colors)
    ax.set_ylabel('Capitulation Rate (%)')
    ax.set_title('Capitulation Rate by Condition')
    ax.set_ylim(0, 105)
    ax.tick_params(axis='x', rotation=15)
    for bar, val in zip(bars, cap_vals):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=9)

    # Subplot 3: Verdict Distribution (stacked bar)
    ax = axes[1, 0]
    verdict_colors = {
        'capitulated': '#B22222',
        'logical_failure': '#FFD700',
        'epistemic_dissonance': '#DAA520',
        'hedged': '#9ACD32',
        'held_firm': '#2E8B57'
    }
    all_verdicts = ['capitulated', 'logical_failure', 'epistemic_dissonance', 'hedged', 'held_firm']
    verdict_df = pd.DataFrame([m['verdict_counts'] for m in all_metrics], index=labels)
    for v in all_verdicts:
        if v not in verdict_df.columns:
            verdict_df[v] = 0
    verdict_df = verdict_df[all_verdicts]
    verdict_norm = verdict_df.div(verdict_df.sum(axis=1), axis=0) * 100
    verdict_norm.plot(kind='bar', stacked=True, ax=ax,
                      color=[verdict_colors[v] for v in all_verdicts])
    ax.set_ylabel('Percentage')
    ax.set_title('Verdict Distribution by Condition')
    ax.legend(title='Verdict', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.tick_params(axis='x', rotation=15)

    # Subplot 4: Capitulations heatmap by pressure level
    ax = axes[1, 1]
    cap_by_cond_level = []
    for m in all_metrics:
        level_counts = defaultdict(int)
        for lvl, ver in m['all_level_verdicts']:
            if ver == 'capitulated':
                level_counts[lvl] += 1
        cap_by_cond_level.append(level_counts)

    levels = list(range(1, 15))
    matrix = []
    for cond_counts in cap_by_cond_level:
        row = [cond_counts.get(lvl, 0) for lvl in levels]
        matrix.append(row)

    im = ax.imshow(matrix, cmap='Reds', aspect='auto')
    ax.set_xticks(np.arange(len(levels)))
    ax.set_xticklabels(levels)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel('Pressure Level')
    ax.set_ylabel('Condition')
    ax.set_title('Capitulations by Pressure Level')
    plt.colorbar(im, ax=ax, label='Number of capitulations')

    fig.suptitle(Title, fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'four_way_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Comparison plot saved to {output_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()

    print("\n=== Numeric Summary ===")
    for label, m in zip(labels, all_metrics):
        print(f"{label}: Break={m['avg_break']:.2f}, Cap Rate={m['cap_rate']:.1f}%")
    print("\nVerdict counts:")
    print(verdict_df)