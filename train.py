"""Train the GPT and log a comparable experiment run.

Every run goes to runs/<timestamp>_<name>_s<seed>/ with:
  config.json   hyperparameters, param count, git commit, notes
  metrics.csv   learning curve (written live, one row per eval)
  summary.json  final score, read by stats.py
  gpt.py        snapshot of the architecture used
  changes.diff  git diff of gpt.py vs HEAD
  best.pt       weights at the best validation score

Score = val bpc (bits per character) on the FULL validation set, evaluated
deterministically, at the best eval point. Lower is better.
"""
import argparse
import csv
import importlib
import json
import math
import os
import shutil
import subprocess
import time
from datetime import datetime

import torch
from torch.nn import functional as F

parser = argparse.ArgumentParser(description='train the GPT and log the run')
parser.add_argument('--name', required=True, help='experiment name, e.g. baseline, rope, gelu')
parser.add_argument('--model', default='gpt', help='model module to train: gpt (gpt.py), gpt_rope (gpt_rope.py), ...')
parser.add_argument('--notes', default='', help='free text describing the change')
parser.add_argument('--seed', type=int, default=1337)
parser.add_argument('--max_iters', type=int, default=None, help='default: value in the model file')
parser.add_argument('--eval_interval', type=int, default=None, help='default: value in the model file')
parser.add_argument('--ckpt', default=None, help='best weights are also copied here (default: <model>.pt)')
args = parser.parse_args()

gpt = importlib.import_module(args.model.removesuffix('.py'))
GPTLanguageModel, get_batch, device = gpt.GPTLanguageModel, gpt.get_batch, gpt.device
train_data, val_data = gpt.train_data, gpt.val_data
args.max_iters = args.max_iters or gpt.max_iters
args.eval_interval = args.eval_interval or gpt.eval_interval
args.ckpt = args.ckpt or f"{gpt.__name__}.pt"
model_file = f"{gpt.__name__}.py"

HPARAMS = ['batch_size', 'block_size', 'n_embd', 'n_head', 'n_layer', 'dropout', 'learning_rate', 'rope_base']


@torch.no_grad()
def full_loss(model, data, max_windows=None, bs=64):
    """Mean cross-entropy over consecutive non-overlapping windows. Deterministic."""
    T = gpt.block_size
    n = (len(data) - 1) // T
    if max_windows:
        n = min(n, max_windows)
    x = data[:n * T].view(n, T)
    y = data[1:n * T + 1].view(n, T)
    total, count = 0.0, 0
    for i in range(0, n, bs):
        xb, yb = x[i:i + bs].to(device), y[i:i + bs].to(device)
        logits, _ = model(xb)
        total += F.cross_entropy(logits.reshape(-1, logits.size(-1)), yb.reshape(-1), reduction='sum').item()
        count += yb.numel()
    return total / count


def evaluate(model):
    model.eval()
    val = full_loss(model, val_data)
    train = full_loss(model, train_data, max_windows=len(val_data) // gpt.block_size)  # same size as val
    model.train()
    return train, val


def git(*cmd):
    try:
        # no check=True: 'git diff --no-index' exits 1 when files differ
        return subprocess.run(['git', *cmd], capture_output=True, text=True).stdout
    except Exception:
        return ''


# ---- run directory ----
started = datetime.now()
run_dir = os.path.join('runs', f"{started:%Y%m%d-%H%M%S}_{args.name}_s{args.seed}")
os.makedirs(run_dir)
shutil.copy(model_file, run_dir)
with open(os.path.join(run_dir, 'changes.diff'), 'w', encoding='utf-8') as f:
    # diff of this model file against the original gpt.py at HEAD
    f.write(git('diff', '--no-index', '--', 'gpt.py', model_file) if model_file != 'gpt.py'
            else git('diff', 'HEAD', '--', 'gpt.py'))

# ---- model ----
torch.manual_seed(args.seed)
model = GPTLanguageModel().to(device)
n_params = sum(p.numel() for p in model.parameters())
optimizer = torch.optim.AdamW(model.parameters(), lr=gpt.learning_rate)

config = {
    'name': args.name, 'model': model_file, 'notes': args.notes, 'seed': args.seed,
    'max_iters': args.max_iters, 'eval_interval': args.eval_interval,
    'params': n_params, 'hparams': {k: getattr(gpt, k, None) for k in HPARAMS},
    'git_commit': git('rev-parse', '--short', 'HEAD').strip(),
    'model_file_uncommitted': bool(git('status', '--porcelain', '--', model_file).strip()),
    'device': torch.cuda.get_device_name(0) if device == 'cuda' else 'cpu',
    'started': started.isoformat(timespec='seconds'),
}
with open(os.path.join(run_dir, 'config.json'), 'w') as f:
    json.dump(config, f, indent=2)

print(f"run: {run_dir} | model: {model_file}")
print(f"{n_params/1e6:.3f}M parameters | {args.max_iters} iters | seed {args.seed}")

# ---- training ----
metrics_file = open(os.path.join(run_dir, 'metrics.csv'), 'w', newline='')
metrics = csv.writer(metrics_file)
metrics.writerow(['step', 'train_time_s', 'wall_time_s', 'train_loss', 'val_loss', 'val_bpc'])

best = {'val_loss': math.inf}
last = {}
train_time = 0.0  # excludes eval time, so eval frequency doesn't distort speed comparisons
wall_start = time.perf_counter()
status = 'completed'


def sync():
    if device == 'cuda':
        torch.cuda.synchronize()


try:
    seg_start = time.perf_counter()
    for step in range(args.max_iters + 1):
        if step % args.eval_interval == 0 or step == args.max_iters:
            sync()
            train_time += time.perf_counter() - seg_start
            tr, va = evaluate(model)
            bpc = va / math.log(2)
            wall = time.perf_counter() - wall_start
            last = {'step': step, 'train_loss': tr, 'val_loss': va, 'val_bpc': bpc}
            metrics.writerow([step, f"{train_time:.2f}", f"{wall:.2f}", f"{tr:.5f}", f"{va:.5f}", f"{bpc:.5f}"])
            metrics_file.flush()
            mark = ''
            if va < best['val_loss']:
                best = {**last, 'train_time_s': train_time}
                torch.save(model.state_dict(), os.path.join(run_dir, 'best.pt'))
                mark = ' *best'
            eta = (train_time / step * (args.max_iters - step)) if step else 0
            print(f"step {step:5d} | train {tr:.4f} | val {va:.4f} | bpc {bpc:.4f} | "
                  f"{train_time:6.1f}s train, {wall:6.1f}s total, eta ~{eta/60:.1f}min{mark}")
            seg_start = time.perf_counter()
        if step == args.max_iters:
            break

        xb, yb = get_batch('train')
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
except KeyboardInterrupt:
    status = 'interrupted'
    print('\ninterrupted - saving summary of what was trained so far')
finally:
    metrics_file.close()

# ---- summary ----
tokens = (last.get('step', 0)) * gpt.batch_size * gpt.block_size
summary = {
    **config,
    'status': status,
    'score_bpc': best.get('val_bpc'),
    'best_val_loss': best.get('val_loss'),
    'best_step': best.get('step'),
    'best_train_time_s': best.get('train_time_s'),
    'final_step': last.get('step'),
    'final_train_loss': last.get('train_loss'),
    'final_val_loss': last.get('val_loss'),
    'final_val_bpc': last.get('val_bpc'),
    'train_time_s': train_time,
    'tokens_per_s': tokens / train_time if train_time else None,
    'run_dir': run_dir,
}
with open(os.path.join(run_dir, 'summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)

if best.get('val_bpc') is not None:
    shutil.copy(os.path.join(run_dir, 'best.pt'), args.ckpt)
    print(f"\nSCORE (best val bpc): {best['val_bpc']:.4f} at step {best['step']} "
          f"| final val bpc {last['val_bpc']:.4f} | train time {train_time/60:.1f}min")
    print(f"best weights copied to {args.ckpt}")
