import importlib.util
import json
import unittest
from pathlib import Path

import numpy as np

from fullgoal_quant import inference

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'sample_results'/'2026-10-16'


def robustness():
    spec=importlib.util.spec_from_file_location('robustness',ROOT/'scripts'/'robustness.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class InferenceTests(unittest.TestCase):
    def test_stationary_indices_are_contiguous_blocks(self):
        index=inference.stationary_indices(50,200,10,np.random.default_rng(1))
        self.assertEqual(index.shape,(200,50))
        self.assertTrue(((index>=0)&(index<50)).all())
        step=(index[:,1:]-index[:,:-1])%50
        # Within a block the index advances by one (circularly); mean block ≈ 10.
        self.assertAlmostEqual(float((step==1).mean()),0.9,delta=0.03)
        with self.assertRaises(ValueError):
            inference.stationary_indices(1,10,5,np.random.default_rng(1))

    def test_block_length_one_matches_iid_resampling_scale(self):
        rng=np.random.default_rng(7)
        returns=rng.normal(0.0005,0.01,500)
        result=inference.bootstrap(returns,samples=4000,mean_block=1,seed=3)
        low,high=result['sharpe_ci']
        self.assertLess(low,result['sharpe'])
        self.assertGreater(high,result['sharpe'])
        # Asymptotic s.e. of an annualised Sharpe near zero is about sqrt(252/n).
        self.assertAlmostEqual((high-low)/2/1.96,(252/500)**0.5,delta=0.08)

    def test_bootstrap_detects_clear_edge_and_pairs_benchmark(self):
        rng=np.random.default_rng(11)
        common=rng.normal(0,0.01,400)
        strategy=common+0.002
        result=inference.bootstrap(strategy,common,samples=2000,mean_block=10,seed=5)
        # Shared noise cancels in the paired excess, so the interval is tight.
        self.assertAlmostEqual(result['excess_annual'],0.002*252)
        self.assertAlmostEqual(result['excess_ci'][0],0.002*252)
        self.assertEqual(result['prob_excess_le_0'],0.0)
        with self.assertRaises(ValueError):
            inference.bootstrap(strategy,common[:-1])

    def test_bootstrap_is_seeded(self):
        returns=np.random.default_rng(2).normal(0,0.01,126)
        self.assertEqual(inference.bootstrap(returns,samples=500,seed=9),inference.bootstrap(returns,samples=500,seed=9))
        self.assertNotEqual(inference.bootstrap(returns,samples=500,seed=9)['sharpe_ci'],
                            inference.bootstrap(returns,samples=500,seed=10)['sharpe_ci'])

    def test_probabilistic_and_deflated_sharpe(self):
        rng=np.random.default_rng(4)
        noise=rng.normal(0,0.01,1000)
        noise-=noise.mean()
        self.assertAlmostEqual(inference.probabilistic_sharpe(noise),0.5,places=6)
        strong=noise+0.002
        self.assertGreater(inference.probabilistic_sharpe(strong),0.99)
        # A higher hurdle can only lower the probability.
        self.assertLess(inference.probabilistic_sharpe(strong,5.0),inference.probabilistic_sharpe(strong,0.0))
        self.assertEqual(inference.probabilistic_sharpe(np.zeros(100)),0.0)

    def test_expected_max_sharpe_grows_with_trials(self):
        sharpes=[-0.9,-0.5,0.1,0.2,0.3]
        self.assertEqual(inference.expected_max_sharpe(sharpes,1),0.0)
        self.assertLess(inference.expected_max_sharpe(sharpes,2),inference.expected_max_sharpe(sharpes,5))
        self.assertLess(inference.expected_max_sharpe(sharpes,5),inference.expected_max_sharpe(sharpes,50))

    def test_effective_trials_bounds(self):
        rng=np.random.default_rng(6)
        base=rng.normal(0,1,300)
        self.assertAlmostEqual(inference.effective_trials(np.vstack([base,base,base])),1.0)
        independent=inference.effective_trials(rng.normal(0,1,(4,5000)))
        self.assertAlmostEqual(independent,4.0,delta=0.15)

    def test_sign_test_exact_values(self):
        self.assertEqual(inference.sign_test([1]*8),{'positive':8,'periods':8,'p_value':1/256})
        self.assertAlmostEqual(inference.sign_test([1,1,1,-1,-1,-1,-1,-1])['p_value'],219/256)
        self.assertEqual(inference.sign_test([0,0])['p_value'],1.0)

    def close(self, got, want, path='root'):
        if isinstance(want,dict):
            self.assertEqual(set(got),set(want),path)
            for key in want:
                self.close(got[key],want[key],f'{path}.{key}')
        elif isinstance(want,list):
            self.assertEqual(len(got),len(want),path)
            for i,item in enumerate(want):
                self.close(got[i],item,f'{path}[{i}]')
        elif isinstance(want,float):
            self.assertAlmostEqual(got,want,places=9,msg=path)
        else:
            self.assertEqual(got,want,path)

    def test_published_robustness_file_reproduces(self):
        module=robustness()
        result=module.analyse(RUN)
        # Tolerance rather than bit equality: summation order may differ by CPU.
        self.close(result,json.loads((RUN/'robustness.json').read_text(encoding='utf-8')))
        self.assertIn('|sector_rotation|',(RUN/'robustness.md').read_text(encoding='utf-8'))
        published=json.loads((RUN/'comparison.json').read_text(encoding='utf-8'))['results']
        for name,item in result['strategies'].items():
            self.assertAlmostEqual(item['sharpe'],published[name]['test']['sharpe'],places=9)
            low,high=item['bootstrap']['10']['sharpe_ci']
            self.assertLess(low,0)
            self.assertGreater(high,0)


if __name__=='__main__':
    unittest.main()
