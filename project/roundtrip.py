"""HF -> Megatron -> HF round trip, with settings that work on Colab (no APEX)."""
import argparse
import os

from rich.console import Console
from megatron.bridge import AutoBridge
from megatron.bridge.models.conversion import weights_verification_table

parser = argparse.ArgumentParser()
parser.add_argument("--hf-model-id", default="Qwen/Qwen3-0.6B")
parser.add_argument("--output-dir", default="/content/drive/MyDrive/megatron_project/roundtrip")
args = parser.parse_args()

console = Console()
bridge = AutoBridge.from_hf_pretrained(args.hf_model_id)

provider = bridge.to_megatron_provider(load_weights=True)   # architecture + "load HF weights"
provider.gradient_accumulation_fusion = False               # needs APEX, which Colab doesn't have
if hasattr(provider, "finalize"):
    provider.finalize()

megatron_model = provider.provide_distributed_model(wrap_with_ddp=False)
console.print(weights_verification_table(bridge, megatron_model))

save_path = os.path.join(args.output_dir, args.hf_model_id.split("/")[-1])
console.print(f"Saving HF checkpoint to {save_path}")
bridge.save_hf_pretrained(megatron_model, save_path)