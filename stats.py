"""Compare experiment runs against the baseline.

  python stats.py                 leaderboard grouped by experiment name
  python stats.py --all           also list every individual run
  python stats.py --ref baseline  choose the reference experiment
  python stats.py --plot          save runs/compare.png with learning curves
"""
import argparse
import csv
import glob
import json
import os
import statistics as st

parser = argparse.ArgumentParser(description='compare experiment runs')
parser.add_argument('--ref', default='baseline', help='reference experiment name')
parser.add_argument('--noise', type=float, default=0.01,
                    help='bpc noise threshold used while the reference has fewer than 2 seeds')
parser.add_argument('--all', action='store_true', help='list every run')
parser.add_argument('--plot', action='store_true', help='save runs/compare.png')
parser.add_argument('--only', nargs='*', help='only these experiment names in the plot')
parser.add_argument('--include_interrupted', action='store_true',
                    help='also count runs stopped with Ctrl+C (not comparable: fewer steps)')
args = parser.parse_args()

runs, skipped = [], []
for path in glob.glob(os.path.join('runs', '*', 'summary.json')):
    with open(path) as f:
        s = json.load(f)
    if s.get('score_bpc') is None:
        continue
    if s.get('status') != 'completed' and not args.include_interrupted:
        skipped.append(s)
        continue
    runs.append(s)
if skipped:
    print(f"(ignoring {len(skipped)} interrupted run(s): {', '.join(os.path.basename(s['run_dir']) for s in skipped)}"
          f" - use --include_interrupted to count them)")

if not runs:
    raise SystemExit('no runs found - run: python train.py --name baseline')

groups = {}
for r in runs:
    groups.setdefault(r['name'], []).append(r)


def agg(rs):
    scores = [r['score_bpc'] for r in rs]
    return {
        'n': len(rs),
        'mean': st.mean(scores),
        'std': st.stdev(scores) if len(scores) > 1 else None,
        'params': rs[-1]['params'],
        'time': st.mean(r['train_time_s'] for r in rs),
        'tps': st.mean(r['tokens_per_s'] or 0 for r in rs),
        'best_step': st.mean(r['best_step'] for r in rs),
        'final': st.mean(r['final_val_bpc'] for r in rs),
        'hparams': rs[-1]['hparams'],
        'interrupted': any(r['status'] != 'completed' for r in rs),
        'iters': {r['max_iters'] for r in rs},
    }


stats = {name: agg(rs) for name, rs in groups.items()}
ref = stats.get(args.ref)
if ref:
    noise = 2 * ref['std'] if ref['std'] is not None else args.noise
    noise_src = (f"2x std of {ref['n']} '{args.ref}' seeds" if ref['std'] is not None
                 else f"--noise default; run '{args.ref}' with more seeds for a real estimate")
else:
    print(f"(no '{args.ref}' runs yet - showing absolute scores only)")

print(f"\nSCORE = best val bpc on the full validation set (lower is better)")
if ref:
    print(f"noise threshold: {noise:.4f} bpc ({noise_src})")
print()
header = f"{'experiment':<22}{'n':>3}{'score':>9}{'±std':>8}{'Δ vs ref':>10}{'Δ%':>7}  {'verdict':<10}" \
         f"{'final':>8}{'best@':>7}{'params':>9}{'time':>8}{'tok/s':>9}"
print(header)
print('-' * len(header))
for name, s in sorted(stats.items(), key=lambda kv: kv[1]['mean']):
    std = f"{s['std']:.4f}" if s['std'] is not None else '-'
    if ref and name != args.ref:
        d = s['mean'] - ref['mean']
        delta, pct = f"{d:+.4f}", f"{100 * d / ref['mean']:+.1f}"
        verdict = 'noise' if abs(d) < noise else ('BETTER' if d < 0 else 'worse')
    else:
        delta, pct, verdict = '', '', 'reference' if name == args.ref else ''
    flags = []
    if s['interrupted']:
        flags.append('interrupted')
    if ref and s['iters'] != ref['iters']:
        flags.append(f"iters={sorted(s['iters'])}")
    if ref and s['hparams'].get('block_size') != ref['hparams'].get('block_size'):
        flags.append('block_size differs')
    print(f"{name:<22}{s['n']:>3}{s['mean']:>9.4f}{std:>8}{delta:>10}{pct:>7}  {verdict:<10}"
          f"{s['final']:>8.4f}{s['best_step']:>7.0f}{s['params']/1e6:>8.2f}M{s['time']/60:>7.1f}m"
          f"{s['tps']:>9.0f}" + (f"  [{', '.join(flags)}]" if flags else ''))

print("\nfinal = val bpc at the last step (if much worse than score, the model is overfitting)")
print("best@ = step of the best score | time = training time excluding evals")

if args.all:
    print()
    for r in sorted(runs, key=lambda r: r['started']):
        print(f"{r['started']}  {r['name']:<20} seed {r['seed']:<6} score {r['score_bpc']:.4f}  "
              f"{r['run_dir']}  {r.get('notes', '')}")

if args.plot:
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        raise SystemExit('matplotlib not installed - run: pip install matplotlib')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    names = args.only or sorted(groups)
    for i, name in enumerate(names):
        color = f"C{i % 10}"
        for j, r in enumerate(groups.get(name, [])):
            with open(os.path.join(r['run_dir'], 'metrics.csv')) as f:
                rows = list(csv.DictReader(f))
            steps = [int(x['step']) for x in rows]
            t = [float(x['train_time_s']) / 60 for x in rows]
            bpc = [float(x['val_bpc']) for x in rows]
            label = name if j == 0 else None
            ax1.plot(steps, bpc, color=color, label=label, alpha=0.8)
            ax2.plot(t, bpc, color=color, label=label, alpha=0.8)
    for ax, xl in ((ax1, 'step'), (ax2, 'training time (min)')):
        ax.set_xlabel(xl)
        ax.set_ylabel('val bpc (lower is better)')
        ax.grid(alpha=0.3)
        ax.legend()
        lows = [s['mean'] for s in stats.values()]
        ax.set_ylim(min(lows) - 0.05, max(lows) + 0.5)
    fig.tight_layout()
    out = os.path.join('runs', 'compare.png')
    fig.savefig(out, dpi=120)
    print(f"\nplot saved to {out}")
