"""Render the Sharpe-interval figure for the README from robustness.json.

Usage: python scripts/plot_intervals.py [sample_results/2026-10-16] [docs/figures]
Needs matplotlib (not a runtime dependency): pip install matplotlib
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RUN = Path(sys.argv[1] if len(sys.argv) > 1 else 'sample_results/2026-10-16')
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else 'docs/figures')
# Same surfaces and inks as plot_holdout.py; one hue because there is one series.
THEMES = {
    'light': {'surface': '#fcfcfb', 'text': '#0b0b0b', 'secondary': '#52514e', 'muted': '#898781',
              'grid': '#e1e0d9', 'mark': '#2a78d6'},
    'dark': {'surface': '#1a1a19', 'text': '#f0efec', 'secondary': '#c3c2b7', 'muted': '#898781',
             'grid': '#2c2c2a', 'mark': '#3987e5'},
}


def draw(result, mode):
    t = THEMES[mode]
    block = str(result['method']['headline_block'])
    rows = sorted(result['strategies'].items(), key=lambda item: item[1]['sharpe'])
    hurdle = result['expected_max_sharpe']['all_trials']
    fig, ax = plt.subplots(figsize=(9, 4.2), dpi=200)
    fig.patch.set_facecolor(t['surface'])
    ax.set_facecolor(t['surface'])
    ax.grid(axis='x', color=t['grid'], linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color(t['grid'])
    ax.tick_params(colors=t['secondary'], labelsize=8, length=0)
    ax.axvline(0, color=t['muted'], linewidth=0.8)
    ax.axvline(hurdle, color=t['muted'], linewidth=0.8, linestyle=(0, (3, 3)))
    for y, (name, item) in enumerate(rows):
        low, high = item['bootstrap'][block]['sharpe_ci']
        ax.plot([low, high], [y, y], color=t['mark'], linewidth=2, solid_capstyle='round')
        ax.scatter(item['sharpe'], y, s=46, color=t['mark'], zorder=3, edgecolors=t['surface'], linewidths=1.5)
        label = f"{item['sharpe']:+.2f}   [{low:+.2f}, {high:+.2f}]".replace('-', '−')
        ax.text(4.0, y, label, color=t['text'], fontsize=8, va='center', family='monospace')
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([name for name, _ in rows], color=t['text'], fontsize=8.5)
    ax.set_xlim(-4, 4)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_xlabel('Annualised Sharpe ratio (rf = 0)', color=t['secondary'], fontsize=8)
    ax.text(4.0, len(rows) - 0.25, 'Sharpe   95% interval', color=t['secondary'], fontsize=8, va='bottom')
    ax.text(hurdle + 0.06, -0.62, f'expected max of {result["trials"]} unskilled: {hurdle:.2f}',
            color=t['secondary'], fontsize=7.5, va='bottom')
    ax.set_title(f"Holdout Sharpe with 95% stationary-bootstrap intervals ({result['sessions']} sessions): "
                 'every interval contains zero', loc='left', color=t['text'], fontsize=10, pad=16)
    fig.subplots_adjust(left=0.15, right=0.74, top=0.87, bottom=0.12)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f'intervals-{mode}.png'
    fig.savefig(path, facecolor=t['surface'])
    plt.close(fig)
    return path


if __name__ == '__main__':
    data = json.loads((RUN / 'robustness.json').read_text(encoding='utf-8'))
    for theme in THEMES:
        print(draw(data, theme))
