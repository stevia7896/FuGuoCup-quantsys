"""Statistical inference on a frozen research run (no backtest is re-run).

Usage: python scripts/robustness.py [sample_results/2026-10-16]
Writes robustness.json and robustness.md next to the run. Deterministic: the
bootstrap seed is fixed, so the same run directory always gives the same file.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from fullgoal_quant import inference

SAMPLES = 10000
SEED = 20261016
BLOCKS = (5, 10, 20)   # mean block length in sessions; 10 is the headline
HEADLINE = 10
BENCHMARK = 'equal_weight'


def daily_returns(run, name):
    # Same convention as backtest.metrics: the first session contributes 0.
    assets = pd.read_csv(run / f'{name}-test-equity.csv')['assets']
    return assets.pct_change().fillna(0.0).to_numpy()


def analyse(run):
    comparison = json.loads((run / 'comparison.json').read_text(encoding='utf-8'))
    names = [n for n in comparison['manifest']['strategies'] if n != 'cash']
    returns = {n: daily_returns(run, n) for n in names}
    sharpes = {n: inference.sharpe(r) for n, r in returns.items()}
    trials = len(names)
    effective = inference.effective_trials(np.vstack([returns[n] for n in names]))
    hurdle = {'all_trials': inference.expected_max_sharpe(list(sharpes.values()), trials),
              'effective_trials': inference.expected_max_sharpe(list(sharpes.values()), effective)}
    strategies = {}
    for name in names:
        benchmark = None if name == BENCHMARK else returns[BENCHMARK]
        by_block = {str(b): inference.bootstrap(returns[name], benchmark, SAMPLES, b, SEED) for b in BLOCKS}
        folds = [f['total_return'] for f in comparison['fold_results'].get(name, [])]
        strategies[name] = {
            'sharpe': sharpes[name],
            'bootstrap': by_block,
            'psr_vs_zero': inference.probabilistic_sharpe(returns[name], 0.0),
            'deflated_sharpe': inference.probabilistic_sharpe(returns[name], hurdle['all_trials']),
            'deflated_sharpe_effective': inference.probabilistic_sharpe(returns[name], hurdle['effective_trials']),
            'fold_sign_test': inference.sign_test(folds) if folds else None,
        }
    return {'run': run.name, 'model_hash': comparison['model_hash'], 'config_hash': comparison['config_hash'],
            'period': comparison['periods']['test'], 'sessions': int(len(returns[BENCHMARK])),
            'method': {'bootstrap': 'stationary (Politis-Romano), paired with benchmark', 'samples': SAMPLES,
                       'seed': SEED, 'mean_block_sessions': list(BLOCKS), 'headline_block': HEADLINE,
                       'interval': '95% percentile', 'benchmark': BENCHMARK},
            'trials': trials, 'effective_trials': effective, 'expected_max_sharpe': hurdle,
            'strategies': strategies}


def render(result):
    block = str(result['method']['headline_block'])
    lines = ['# 统计推断（基于已冻结结果，未重跑回测）', '',
             f"测试期 {result['period'][0]} → {result['period'][1]}，{result['sessions']} 个交易日；"
             f"平稳自助法 {result['method']['samples']} 次，平均块长 {block} 日，种子 {result['method']['seed']}。", '',
             '|策略|夏普|夏普 95% CI|P(夏普≤0)|年化超额 vs equal_weight|超额 95% CI|PSR(>0)|DSR|窗口符号检验 p|',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name, s in result['strategies'].items():
        b = s['bootstrap'][block]
        excess = f"{b['excess_annual']:+.1%}" if 'excess_annual' in b else '—'
        excess_ci = f"[{b['excess_ci'][0]:+.1%}, {b['excess_ci'][1]:+.1%}]" if 'excess_ci' in b else '—'
        fold = s['fold_sign_test']
        sign = f"{fold['p_value']:.2f}（{fold['positive']}/{fold['periods']}）" if fold else '无法验证'
        lines.append(f"|{name}|{s['sharpe']:.2f}|[{b['sharpe_ci'][0]:.2f}, {b['sharpe_ci'][1]:.2f}]|"
                     f"{b['prob_sharpe_le_0']:.2f}|{excess}|{excess_ci}|{s['psr_vs_zero']:.2f}|"
                     f"{s['deflated_sharpe']:.2f}|{sign}|")
    h = result['expected_max_sharpe']
    lines += ['', f"多重比较：{result['trials']} 个非现金策略，按日收益平均相关性折算的有效试验数为 "
              f"{result['effective_trials']:.2f}。无技能假设下最大夏普的期望为 {h['all_trials']:.2f}"
              f"（有效试验数口径 {h['effective_trials']:.2f}）；DSR 为真实夏普超过该门槛的概率。", '',
              '块长敏感性（夏普 95% CI）：', '', '|策略|' + '|'.join(f'块长 {b}' for b in result['method']['mean_block_sessions']) + '|',
              '|---|' + '---:|' * len(result['method']['mean_block_sessions'])]
    for name, s in result['strategies'].items():
        lines.append(f'|{name}|' + '|'.join(
            f"[{s['bootstrap'][str(b)]['sharpe_ci'][0]:.2f}, {s['bootstrap'][str(b)]['sharpe_ci'][1]:.2f}]"
            for b in result['method']['mean_block_sessions']) + '|')
    lines += ['', '局限：测试期已被查看，以上区间描述的是该区间内的抽样不确定性，不是未见样本上的预测区间；'
              '126 日样本只包含一次行情，自助法无法生成样本中没有出现过的市场状态。', '']
    return '\n'.join(lines)


if __name__ == '__main__':
    directory = Path(sys.argv[1] if len(sys.argv) > 1 else 'sample_results/2026-10-16')
    output = analyse(directory)
    (directory / 'robustness.json').write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (directory / 'robustness.md').write_text(render(output), encoding='utf-8')
    print(render(output))
