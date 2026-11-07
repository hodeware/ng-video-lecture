import argparse
import os
import torch
from gpt import GPTLanguageModel, device, encode, decode, stoi

parser = argparse.ArgumentParser(description='generate text from saved GPT weights')
parser.add_argument('--ckpt', default='gpt.pt', help='weights file produced by train.py')
parser.add_argument('--tokens', type=int, default=500, help='characters to generate per sample')
parser.add_argument('--samples', type=int, default=1, help='number of samples')
parser.add_argument('--prompt', default='', help='starting text')
parser.add_argument('--seed', type=int, default=None, help='seed for reproducible output (default: random)')
parser.add_argument('--out', default=None, help='also write the samples to this file')
args = parser.parse_args()

if not os.path.exists(args.ckpt):
    raise SystemExit(f"{args.ckpt} not found - run train.py first")

model = GPTLanguageModel().to(device)
model.load_state_dict(torch.load(args.ckpt, map_location=device))
model.eval()

seed = args.seed if args.seed is not None else int.from_bytes(os.urandom(4), 'little')
torch.manual_seed(seed)
print(f"seed: {seed}")

unknown = sorted(set(args.prompt) - set(stoi))
if unknown:
    raise SystemExit(f"prompt contains characters not in the vocabulary: {unknown}")
start = encode(args.prompt) if args.prompt else [0]

results = []
for i in range(args.samples):
    context = torch.tensor([start], dtype=torch.long, device=device)
    with torch.no_grad():
        text = decode(model.generate(context, max_new_tokens=args.tokens)[0].tolist())
    results.append(text)
    print(f"\n===== sample {i+1}/{args.samples} =====")
    print(text)

if args.out:
    with open(args.out, 'w', encoding='utf-8') as f:
        f.write('\n\n=====\n\n'.join(results))
    print(f"\nwritten to {args.out}")
