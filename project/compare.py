"""Same prompt through Hugging Face and Megatron: compare logits, then generate code with both."""
import argparse

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from megatron.bridge import AutoBridge

parser = argparse.ArgumentParser()
parser.add_argument("--hf-model-id", default="Qwen/Qwen2.5-Coder-1.5B")
parser.add_argument("--prompt", default="def fibonacci(n):")
parser.add_argument("--new-tokens", type=int, default=48)
parser.add_argument("--prompt-file", default=None)      # add above args = parser.parse_args()
args = parser.parse_args()
if args.prompt_file:                                     # add these two lines below it
    args.prompt = open(args.prompt_file).read()[:8000]
args = parser.parse_args()

tok = AutoTokenizer.from_pretrained(args.hf_model_id)
ids = tok(args.prompt, return_tensors="pt").input_ids.cuda()

# --- Megatron model (same recipe as roundtrip.py) ---
bridge = AutoBridge.from_hf_pretrained(args.hf_model_id)
provider = bridge.to_megatron_provider(load_weights=True)
provider.gradient_accumulation_fusion = False          # no APEX on Colab
provider.finalize()
mg = provider.provide_distributed_model(wrap_with_ddp=False)[0].eval()

def megatron_logits(x):
    pos = torch.arange(x.size(1), device=x.device).unsqueeze(0)
    out = mg(input_ids=x, position_ids=pos, attention_mask=None)   # None = Megatron builds the causal mask
    if isinstance(out, tuple):
        out = out[0]
    if out.shape[0] != x.shape[0]:      # Megatron's internal layout is [seq, batch, vocab]
        out = out.transpose(0, 1)
    return out.float()

# --- Hugging Face model ---
hf = AutoModelForCausalLM.from_pretrained(args.hf_model_id, dtype=torch.bfloat16).cuda().eval()

def hf_logits(x):
    return hf(x).logits.float()

# --- 1. One forward pass: compare every logit ---
with torch.no_grad():
    a = hf_logits(ids)
    b = megatron_logits(ids)[..., : a.size(-1)]   # Megatron may pad the vocab; drop the padding

diff = (a - b).abs()
print(f"\nPrompt: {args.prompt!r}  ({ids.size(1)} tokens)   logits shape {tuple(a.shape)}")
print(f"max |HF - Megatron|  = {diff.max().item():.4f}")
print(f"mean |HF - Megatron| = {diff.mean().item():.5f}")
print(f"cosine similarity (last position) = {F.cosine_similarity(a[0, -1], b[0, -1], dim=0).item():.6f}")
print(f"same top-1 token at every position: {(a.argmax(-1) == b.argmax(-1)).float().mean().item():.0%}")
for name, lg in (("HF", a), ("Megatron", b)):
    top5 = lg[0, -1].topk(5).indices.tolist()
    print(f"{name:9s} top-5 next tokens: {[tok.decode([t]) for t in top5]}")

# --- 2. Greedy generation with each model ---
def generate(logits_fn):
    x = ids.clone()
    with torch.no_grad():
        for _ in range(args.new_tokens):
            nxt = logits_fn(x)[:, -1].argmax(-1, keepdim=True)
            x = torch.cat([x, nxt], dim=1)
            if nxt.item() == tok.eos_token_id:
                break
    return tok.decode(x[0])

print("\n===== HF generation =====\n" + generate(hf_logits))
print("\n===== Megatron generation =====\n" + generate(megatron_logits))