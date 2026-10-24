"""Load reviewed official source without executing upstream training/config scripts.

MPLMM imports are namespaced (only import paths change). G2D's two pure
loss functions are compiled from their original AST; their bodies are unchanged.
"""
import ast
import sys
import types
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1] / "third_party"


def _load(name, path, rename_modules=False):
    if name in sys.modules:
        return sys.modules[name]
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, str(path))
    if rename_modules:
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("modules."):
                node.module = "_icc_mplmm." + node.module[len("modules."):]
    module = types.ModuleType(name)
    module.__file__ = str(path)
    module.__package__ = name.rpartition(".")[0]
    sys.modules[name] = module
    exec(compile(tree, str(path), "exec"), module.__dict__)
    return module


@lru_cache(None)
def mplmm_source():
    package = types.ModuleType("_icc_mplmm")
    package.__path__ = [str(ROOT / "MPLMM/modules")]
    sys.modules[package.__name__] = package
    for filename in ("multihead_attention", "position_embedding", "transformer"):
        _load(f"_icc_mplmm.{filename}", ROOT / f"MPLMM/modules/{filename}.py", True)
    return _load("_icc_mplmm.model", ROOT / "MPLMM/src/model.py", True)


def mplmm_config(dims, hidden=256, seq_len=(1, 1, 1), heads=4):
    # MPLMM order is language, AUDIO, VISUAL; public ICC order is T,V,A.
    return SimpleNamespace(orig_d_l=dims[0], orig_d_a=dims[2], orig_d_v=dims[1],
        proj_dim=hidden, prompt_dim=hidden, prompt_length=2, num_heads=heads,
        layers=1, attn_dropout=0., attn_dropout_a=0., attn_dropout_v=0.,
        relu_dropout=0., res_dropout=0., out_dropout=0., embed_dropout=0.,
        attn_mask=False, seq_len=seq_len, output_dim=2)


@lru_cache(None)
def g2d_losses():
    path = ROOT / "G2D/student.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                 and n.name in {"distillation_loss", "feature_loss"}]
    if len(functions) != 2:
        raise RuntimeError("Official G2D loss source changed")
    namespace = {"nn": nn, "torch": torch}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)
    return SimpleNamespace(**{n.name: namespace[n.name] for n in functions})


def mplmm_missing_modes(mask):
    # visible TVA bit codes -> official missing_mode: VA, TV, TA, V, A, T, TVA
    table = torch.tensor([-1, 4, 3, 0, 5, 2, 1, 6], device=mask.device)
    code = (mask.long() * mask.new_tensor([4, 2, 1], dtype=torch.long)).sum(1)
    return table[code]
