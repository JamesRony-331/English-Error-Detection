"""将 safetensors 转为纯 PyTorch state_dict，并逐张量校验。"""
import argparse
from pathlib import Path
import torch
from safetensors.torch import load_file

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('target', type=Path)
    args = parser.parse_args()
    if args.target.exists():
        parser.error('目标已存在，请换一个文件名，避免覆盖')
    state = load_file(str(args.source), device='cpu')
    args.target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, args.target)
    restored = torch.load(args.target, map_location='cpu', weights_only=True)
    assert state.keys() == restored.keys()
    assert all(torch.equal(state[key], restored[key]) for key in state)
    print(f'已转换并验证 {len(state)} 个张量：{args.target}')
