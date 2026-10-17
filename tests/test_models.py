"""CPU architecture/loader checks using reference SDPA, not CUDA kernel validation."""
import importlib
import sys
from types import ModuleType
import pytest
import torch
from torch import nn
import torch.nn.functional as F
from safetensors.torch import save_file
from transformers import Qwen2Config, Qwen3Config, LlamaConfig, AutoModelForCausalLM
from nanovllm.models.registry import validate_model_config, build_model
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.utils.context import set_context, reset_context
from nanovllm.utils.loader import load_model


SCALING = dict(rope_type="llama3", factor=32., low_freq_factor=1., high_freq_factor=4.,
               original_max_position_embeddings=32)


def config_for(family):
    cls = {"qwen2": Qwen2Config, "qwen3": Qwen3Config, "llama": LlamaConfig}[family]
    return cls(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8, vocab_size=64,
        max_position_embeddings=128, dtype="float16", tie_word_embeddings=True,
        rope_scaling=SCALING.copy() if family == "llama" else None)


@pytest.fixture
def cpu_backend(monkeypatch):
    class ReferenceAttention(nn.Module):
        def __init__(self, heads, head_dim, scale, kv_heads):
            super().__init__()
            self.scale = scale
            self.groups = heads // kv_heads

        def forward(self, q, k, v):
            from nanovllm.utils.context import get_context
            context = get_context()
            if context.slot_mapping is not None:
                self.k_cache.view(-1, *k.shape[1:])[context.slot_mapping] = k
                self.v_cache.view(-1, *v.shape[1:])[context.slot_mapping] = v
            if context.cu_seqlens_k is None:
                qs, ks = [0, q.size(0)], [0, q.size(0)]
            else:
                qs, ks = context.cu_seqlens_q.tolist(), context.cu_seqlens_k.tolist()
            outputs = []
            for i, (start, end) in enumerate(zip(qs, qs[1:])):
                qi = q[start:end]
                nk = ks[i+1] - ks[i]
                if context.block_tables is None:
                    ki, vi = k[start:end], v[start:end]
                else:
                    table = context.block_tables[i]
                    table = table[table >= 0]
                    ki = self.k_cache[table].flatten(0, 1)[:nk]
                    vi = self.v_cache[table].flatten(0, 1)[:nk]
                qi = qi.transpose(0, 1).unsqueeze(0)
                ki = ki.repeat_interleave(self.groups, dim=1).transpose(0, 1).unsqueeze(0)
                vi = vi.repeat_interleave(self.groups, dim=1).transpose(0, 1).unsqueeze(0)
                # Independent reference mask for bottom-right causal alignment.
                mask = torch.arange(nk)[None, :] <= torch.arange(nk-(end-start), nk)[:, None]
                out = F.scaled_dot_product_attention(qi, ki, vi, attn_mask=mask, scale=self.scale)
                outputs.append(out[0].transpose(0, 1))
            return torch.cat(outputs)

    stub = ModuleType("nanovllm.layers.attention")
    stub.Attention = ReferenceAttention
    monkeypatch.setitem(sys.modules, "nanovllm.layers.attention", stub)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 1)
    sys.modules.pop("nanovllm.models.qwen3", None)
    get_rope.cache_clear()
    yield
    sys.modules.pop("nanovllm.models.qwen3", None)
    get_rope.cache_clear()
    reset_context()


@pytest.mark.parametrize("family", ["qwen2", "qwen3", "llama"])
def test_dense_model_logits_match_hf_with_reference_attention(cpu_backend, tmp_path, family):
    torch.manual_seed(3)
    config = config_for(family)
    reference = AutoModelForCausalLM.from_config(config).float().eval()
    weights = {name: value.detach().clone() for name, value in reference.state_dict().items()}
    # Emulate a tied checkpoint which stores only embedding weights.
    weights.pop("lm_head.weight")
    save_file(weights, tmp_path / "model.safetensors")
    model = build_model(config).eval()
    load_model(model, str(tmp_path))
    tokens = torch.tensor([1, 8, 5, 3, 12, 7])
    with torch.no_grad():
        expected = reference(tokens.unsqueeze(0)).logits[0, -1:]
        set_context(True, cu_seqlens_q=torch.tensor([0, len(tokens)]))
        actual = model.compute_logits(model(tokens, torch.arange(len(tokens))))
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    attn = model.model.layers[0].self_attn
    assert hasattr(attn, "q_norm") == (family == "qwen3")
    assert (attn.qkv_proj.bias is not None) == (family == "qwen2")


def test_llama3_rope_matches_transformers_reference():
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
    from nanovllm.layers.rotary_embedding import RotaryEmbedding
    config = config_for("llama")
    inv, _ = ROPE_INIT_FUNCTIONS["llama3"](config, torch.device("cpu"))
    rope = RotaryEmbedding(8, 8, 128, config.rope_theta, tuple(sorted(SCALING.items())))
    pos = torch.arange(128.)
    freq = torch.outer(pos, inv)
    torch.testing.assert_close(rope.cos_sin_cache[:, 0], torch.cat((freq.cos(), freq.sin()), -1))


@pytest.mark.parametrize("change,match", [
    ({"model_type": "qwen3_5"}, "unsupported model_type"),
    ({"quantization_config": {"quant_method": "awq"}}, "quantized"),
    ({"use_sliding_window": True}, "sliding-window"),
    ({"rope_scaling": {"rope_type": "yarn"}}, "RoPE"),
    ({"hidden_act": "gelu"}, "SiLU"),
    ({"mlp_bias": True}, "MLP bias"),
    ({"attention_bias": True}, "output bias"),
    ({"architectures": ["Qwen3MoeForCausalLM"]}, "architectures"),
])
def test_unsupported_configs_fail_before_gpu_init(change, match):
    config = config_for("qwen3")
    for key, value in change.items():
        setattr(config, key, value)
    with pytest.raises(ValueError, match=match):
        validate_model_config(config)


def test_missing_checkpoint_tensor_is_detected(cpu_backend, tmp_path):
    config = config_for("qwen2")
    reference = AutoModelForCausalLM.from_config(config)
    weights = {name: value.detach().clone() for name, value in reference.state_dict().items()}
    weights.pop("model.layers.0.self_attn.k_proj.weight")
    save_file(weights, tmp_path / "model.safetensors")
    with pytest.raises(ValueError, match="k_proj.weight"):
        load_model(build_model(config), str(tmp_path))


@pytest.mark.parametrize("family", ["qwen2", "qwen3", "llama"])
@pytest.mark.parametrize("policy", ["decode_first", "prefill_first"])
@pytest.mark.parametrize("blocks", [5, 32])
def test_chunked_cached_generation_matches_hf(cpu_backend, tmp_path, monkeypatch, family, policy, blocks):
    from types import SimpleNamespace
    from nanovllm import SamplingParams
    from nanovllm.engine.sequence import Sequence
    from nanovllm.engine.scheduler import Scheduler
    from nanovllm.engine.batch import build_prefill_inputs
    monkeypatch.setattr(Sequence, "block_size", 4)
    torch.manual_seed(21)
    config = config_for(family)
    reference = AutoModelForCausalLM.from_config(config).float().eval()
    save_file({k: v.detach().clone() for k, v in reference.state_dict().items()}, tmp_path / "model.safetensors")
    model = build_model(config).eval()
    load_model(model, str(tmp_path))
    for layer in model.model.layers:
        layer.self_attn.attn.k_cache = torch.empty(blocks, 4, 2, 8)
        layer.self_attn.attn.v_cache = torch.empty(blocks, 4, 2, 8)
    scheduler = Scheduler(SimpleNamespace(max_num_seqs=8, max_num_batched_tokens=3,
        eos=-1, kvcache_block_size=4, num_kvcache_blocks=blocks, scheduling_policy=policy))
    prompts = [[1, 5], [1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 5, 6, 7, 8, 10]]
    queries = [Sequence(p, SamplingParams(temperature=0, max_tokens=5, ignore_eos=True)) for p in prompts]
    for query in queries:
        scheduler.add(query)
    with torch.inference_mode():
        for _ in range(100):
            if scheduler.is_finished():
                break
            batch, prefill = scheduler.schedule()
            packed = build_prefill_inputs(batch, 4)
            set_context(True, torch.tensor(packed.cu_seqlens_q), torch.tensor(packed.cu_seqlens_k),
                packed.max_seqlen_q, packed.max_seqlen_k, torch.tensor(packed.slot_mapping),
                None, torch.tensor(packed.block_tables) if packed.block_tables else None)
            logits = model.compute_logits(model(torch.tensor(packed.input_ids), torch.tensor(packed.positions)))
            tokens = [int(logits[i].argmax()) if q.num_cached_tokens + q.num_scheduled_tokens == len(q)
                      else None for i, q in enumerate(batch)]
            scheduler.postprocess(batch, tokens, prefill)
        assert scheduler.is_finished()
        for prompt, query in zip(prompts, queries):
            expected = list(prompt)
            for _ in range(5):
                expected.append(int(reference(torch.tensor([expected])).logits[0, -1].argmax()))
            assert query.completion_token_ids == expected[len(prompt):]
        assert not scheduler.block_manager.used_block_ids
