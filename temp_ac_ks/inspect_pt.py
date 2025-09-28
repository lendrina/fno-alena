import torch
import sys
from pathlib import Path

def inspect_pt_file(filename: str):
    path = Path(filename)
    if not path.exists():
        print(f"❌ File not found: {filename}")
        return

    print(f"🔍 Inspecting file: {path.resolve()}")
    try:
        # Load to CPU only (avoid GPU OOM)
        obj = torch.load(path, map_location="cpu")
    except Exception as e:
        print(f"⚠️ Could not fully load file: {e}")
        return

    print(f"\n✅ Top-level object type: {type(obj)}")

    if isinstance(obj, dict):
        print(f"Dictionary with {len(obj)} keys. Showing up to 20:")
        for k in list(obj.keys())[:20]:
            v = obj[k]
            if isinstance(v, torch.Tensor):
                print(f"  {k}: Tensor shape {tuple(v.shape)}, dtype={v.dtype}")
            else:
                print(f"  {k}: {type(v)}")
    elif isinstance(obj, (list, tuple)):
        print(f"{type(obj).__name__} of length {len(obj)}. Showing first 5:")
        for i, v in enumerate(obj[:5]):
            if isinstance(v, torch.Tensor):
                print(f"  [{i}]: Tensor shape {tuple(v.shape)}, dtype={v.dtype}")
            else:
                print(f"  [{i}]: {type(v)}")
    elif isinstance(obj, torch.Tensor):
        print(f"Tensor of shape {tuple(obj.shape)}, dtype={obj.dtype}")
    else:
        print(f"Other object type: {type(obj)}")
        print(obj)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_pt.py <path_to_file.pt>")
    else:
        inspect_pt_file(sys.argv[1])