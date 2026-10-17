import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def load_model(model: nn.Module, path: str):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    files = sorted(glob(os.path.join(path, "*.safetensors")))
    if not files:
        raise ValueError(f"no safetensors checkpoint files in {path}")
    loaded = set()
    for file in files:
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                if weight_name.endswith(".rotary_emb.inv_freq"):
                    continue  # Reconstructed non-persistent RoPE buffers.
                if weight_name in loaded:
                    raise ValueError(f"duplicate checkpoint tensor: {weight_name}")
                loaded.add(weight_name)
                for k in packed_modules_mapping:
                    if k in weight_name:
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)
                        param = model.get_parameter(param_name)
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        break
                else:
                    param = model.get_parameter(weight_name)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name))
    expected = set()
    for name, _ in model.named_parameters():
        for packed_name, parts in (("qkv_proj", ("q_proj", "k_proj", "v_proj")),
                                   ("gate_up_proj", ("gate_proj", "up_proj"))):
            if packed_name in name:
                expected.update(name.replace(packed_name, part) for part in parts)
                break
        else:
            expected.add(name)
    if model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr():
        expected.discard("lm_head.weight")
    missing = expected - loaded
    if missing:
        raise ValueError(f"checkpoint is missing tensors: {sorted(missing)}")
