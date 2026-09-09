"""Plot an analytically derived rational SSX counterexample and actual output.

Usage: python examples/ssx/ssx5_rational_audit_figure.py --output-dir /tmp/ssx-audit
"""
from pathlib import Path
import argparse
import json
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    from mmcore.nurbs._nurbs_eval import NURBSSurfaceTuple
    from mmcore.numeric.intersection.ssx._nssx5 import nurbs_ssx
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    epsilon = 8e-6
    cp = np.array([[[x, y, z] for y in (-1., 1.)]
                   for x, z in zip((-1., 0., 1.), (1., -1., 1.))])
    weights = np.ones((3, 2))
    weights[1] += epsilon
    knots = np.array([0., 0., 1., 1.])
    surface = NURBSSurfaceTuple(3, 2, np.array([0., 0., 0., 1., 1., 1.]),
                                knots, cp, weights)
    plane_cp = np.array([[[x, y, 0.] for y in (-1., 1.)] for x in (-1., 1.)])
    plane = NURBSSurfaceTuple(2, 2, knots, knots, plane_cp, np.ones((2, 2)))
    result = nurbs_ssx(surface, plane, atol=1e-5, max_cells=15000)
    expected = np.sqrt(epsilon*(2.+epsilon))/(2.*(1.+epsilon))
    if not result['complete'] or len(result['branches']) != 2:
        raise RuntimeError(f"Rational audit did not complete with two branches: {result['status']}")
    positions = sorted(float(np.asarray(b.curve[1])[:, 0].mean()) for b in result['branches'])
    np.testing.assert_allclose(positions, [-expected, expected], atol=1e-9, rtol=0.)

    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.4), layout='constrained')
    parameter = np.linspace(.497, .503, 1201)
    numerator = (1.-parameter)**2-2.*(1.+epsilon)*parameter*(1.-parameter)+parameter**2
    denominator = 1.+2.*epsilon*parameter*(1.-parameter)
    x = (2.*parameter-1.)/denominator
    axes[0].plot(1000.*x, 1e6*numerator/denominator, color='#147d92', lw=2,
                 label='Supplied rational surface')
    axes[0].plot(1000.*(2.*parameter-1.), 1e6*(2.*parameter-1.)**2,
                 color='#b15e3c', ls='--', label='After rounding weights to one')
    axes[0].axhline(0., color='#9aa1a8', lw=.9)
    axes[0].scatter(1000.*np.array([-expected, expected]), [0., 0.],
                    color='#147d92', zorder=4)
    axes[0].set(xlabel='x × 1,000', ylabel='z × 1,000,000',
                title='Representation changes the zero set', ylim=(-6., 20.))
    axes[0].legend(loc='upper center', fontsize=8, frameon=False)

    for i, branch in enumerate(result['branches']):
        xyz = np.asarray(branch.curve[1])
        axes[1].plot(1000.*xyz[:, 0], xyz[:, 1], color='#147d92', lw=2.4,
                     label='Actual NURBS SSX output' if i == 0 else None)
    axes[1].plot([0., 0.], [-1., 1.], color='#b15e3c', ls='--',
                 label='Incorrect polynomial tangency')
    axes[1].set(xlabel='x × 1,000', ylabel='y', xlim=(-3., 3.), ylim=(-1.08, 1.08),
                title='Two transverse intersection branches')
    axes[1].legend(loc='lower center', fontsize=8, frameon=False)
    fig.suptitle('A weight perturbation of 0.000008 splits a double root', fontsize=14)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ('png', 'svg'):
        fig.savefig(args.output_dir / f'rational-branch-preservation.{suffix}', dpi=180)
    (args.output_dir / 'rational-branch-preservation.json').write_text(json.dumps({
        'epsilon': epsilon, 'atol': 1e-5, 'expected_x': [-expected, expected],
        'observed_x': positions, 'complete': result['complete'], 'status': result['status'],
    }, indent=2) + '\n')


if __name__ == '__main__':
    main()
