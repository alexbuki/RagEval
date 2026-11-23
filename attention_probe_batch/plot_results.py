"""Create three exploratory figures from attention_probe_batch experiment CSVs."""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def read_csv(path):
    with open(path, newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def make_figures(results_dir):
    results_dir = Path(results_dir)
    scores = read_csv(results_dir / 'scores.csv')
    attention = read_csv(results_dir / 'attention.csv')
    by_case = defaultdict(dict)
    for record in scores:
        by_case[record['id']][record['condition']] = record
    ids = sorted(by_case, key=lambda x: (not x.isdigit(), int(x) if x.isdigit() else x))

    # Plot 1: layerwise attention for AB and BA in the first example.
    case_id = ids[0]
    fig, ax = plt.subplots(figsize=(9, 5))
    for condition in ('AB', 'BA'):
        records = sorted((r for r in attention if r['id'] == case_id and r['condition'] == condition),
                         key=lambda r: int(r['layer']))
        if not records:
            continue
        ax.plot([int(r['layer']) for r in records], [float(r['a_share']) for r in records],
                marker='.', label=f'{condition}: share to A')
    ax.set(title=f'Question {case_id}: attention to Source A among A/B tokens',
           xlabel='Transformer layer (zero-indexed)', ylabel='A share of attention to A and B')
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(alpha=0.25)
    fig.text(0.5, 0.01, 'Measured at final prompt token; descriptive, not causal.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, 1))
    fig.savefig(results_dir / 'figure1_attention_layers.png', dpi=180)
    plt.close(fig)

    # Plot 2: candidate A preference across evidence conditions.
    conditions = ('A', 'B', 'AB', 'BA')
    width = 0.19
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, condition in enumerate(conditions):
        values = [float(by_case[q][condition]['normalized_pair_score_A']) for q in ids]
        ax.bar([j+(i-1.5)*width for j in range(len(ids))], values, width=width, label=condition)
    ax.axhline(0.5, linestyle='--', linewidth=1, label='Equal candidate score')
    ax.set_xticks(range(len(ids)), [f'Q{x}' for x in ids])
    ax.set_ylim(0, 1)
    ax.set(xlabel='ConfRAG question', ylabel='Normalized candidate-A score',
           title='Effect of evidence availability and order on candidate scoring')
    ax.legend(ncol=5, fontsize=8)
    ax.grid(axis='y', alpha=0.2)
    fig.tight_layout()
    fig.savefig(results_dir / 'figure2_candidate_scores.png', dpi=180)
    plt.close(fig)

    # Plot 3: descriptive comparison of attention and preference for both-source trials.
    fig, ax = plt.subplots(figsize=(9, 5))
    for condition in ('AB', 'BA'):
        xx, yy = [], []
        for q in ids:
            records = [r for r in attention if r['id'] == q and r['condition'] == condition and r['a_share']]
            if not records:
                continue
            x = sum(float(r['a_share']) for r in records) / len(records)
            y = float(by_case[q][condition]['normalized_pair_score_A'])
            xx.append(x)
            yy.append(y)
            ax.annotate(f'Q{q}', (x, y), xytext=(4, 4), textcoords='offset points', fontsize=8)
        ax.scatter(xx, yy, label=condition, s=50)
    ax.axhline(0.5, linestyle='--', linewidth=1)
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel='Average layerwise share of A among A/B attention',
           ylabel='Normalized candidate-A score',
           title='Attention and candidate preference (descriptive only)')
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(results_dir / 'figure3_attention_vs_preference.png', dpi=180)
    plt.close(fig)

    print(f'Created 3 figures in {results_dir.resolve()}')
    print(f'Questions included: {", ".join(ids)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', default='results/confrag')
    args = parser.parse_args()
    make_figures(args.results_dir)
