from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect LoRA adapter tensors for visual merger coverage.")
    parser.add_argument("--adapter-path", type=str, default="checkpoints/sft_qwen3vl8b_lora_merger_v1")
    return parser.parse_args()


def resolve_path(path_text: str | Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else (ROOT / path).resolve()


def main() -> None:
    args = parse_args()
    adapter_path = resolve_path(args.adapter_path)
    tensor_path = adapter_path / "adapter_model.safetensors"
    if not tensor_path.exists():
        raise FileNotFoundError(f"adapter_model.safetensors not found: {tensor_path}")

    try:
        from safetensors import safe_open  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError("safetensors is required to inspect adapter tensors.") from exc

    with safe_open(tensor_path, framework="pt", device="cpu") as handle:
        tensor_names = list(handle.keys())

    language_tensors = [name for name in tensor_names if ".language_model." in name or "language_model." in name]
    visual_tensors = [name for name in tensor_names if ".visual." in name or "visual." in name]
    visual_merger_tensors = [name for name in visual_tensors if ".visual.merger." in name or "visual.merger." in name]
    visual_deepstack_tensors = [
        name for name in visual_tensors if ".visual.deepstack_merger_list." in name or "visual.deepstack_merger_list." in name
    ]
    visual_blocks_tensors = [name for name in visual_tensors if ".visual.blocks." in name or "visual.blocks." in name]

    print(f"total adapter tensors: {len(tensor_names)}")
    print(f"language adapter tensors: {len(language_tensors)}")
    print(f"visual adapter tensors: {len(visual_tensors)}")
    print(f"visual.merger adapter tensors: {len(visual_merger_tensors)}")
    print(f"visual.deepstack_merger_list adapter tensors: {len(visual_deepstack_tensors)}")
    print(f"visual.blocks adapter tensors: {len(visual_blocks_tensors)}")
    print(f"First {min(50, len(visual_tensors))} visual adapter tensors:")
    for tensor_name in visual_tensors[:50]:
        print(tensor_name)

    if not visual_tensors:
        raise RuntimeError("No visual adapter tensors found.")
    if len(visual_merger_tensors) + len(visual_deepstack_tensors) == 0:
        raise RuntimeError("No visual merger/deepstack merger adapter tensors found.")


if __name__ == "__main__":
    main()
