"""A deliberately bounded dense-text compatibility matrix."""

SUPPORTED_MODEL_TYPES = {"qwen3", "qwen2", "llama"}


def validate_model_config(config, tensor_parallel_size=1):
    model_type = getattr(config, "model_type", None)
    if model_type not in SUPPORTED_MODEL_TYPES:
        raise ValueError(f"unsupported model_type {model_type!r}; expected dense qwen3/qwen2/llama")
    expected = {"qwen3": "Qwen3ForCausalLM", "qwen2": "Qwen2ForCausalLM", "llama": "LlamaForCausalLM"}
    architectures = getattr(config, "architectures", None)
    if architectures and architectures != [expected[model_type]]:
        raise ValueError(f"unsupported architectures: {architectures}")
    if getattr(config, "quantization_config", None):
        raise ValueError("quantized checkpoints are not supported")
    if getattr(config, "use_sliding_window", False) or "sliding_attention" in (getattr(config, "layer_types", None) or []):
        raise ValueError("sliding-window attention is not supported")
    if getattr(config, "hidden_act", None) != "silu":
        raise ValueError("only SiLU gated MLP is supported")
    if getattr(config, "mlp_bias", False):
        raise ValueError("MLP bias is not supported")
    if model_type != "qwen2" and getattr(config, "attention_bias", False):
        raise ValueError("attention output bias is not supported")
    scaling = getattr(config, "rope_scaling", None) or {}
    rope_type = scaling.get("rope_type", scaling.get("type", "default"))
    if rope_type not in ("default", "llama3"):
        raise ValueError(f"unsupported RoPE scaling {rope_type!r}")
    if rope_type == "llama3":
        required = ("factor", "low_freq_factor", "high_freq_factor", "original_max_position_embeddings")
        if any(k not in scaling for k in required):
            raise ValueError("llama3 RoPE requires factor, frequency factors and original context length")
        if not (scaling["factor"] >= 1 and 0 < scaling["low_freq_factor"] < scaling["high_freq_factor"]
                and scaling["original_max_position_embeddings"] > 0):
            raise ValueError("invalid llama3 RoPE scaling values")
    for name in ("num_attention_heads", "num_key_value_heads", "intermediate_size", "vocab_size"):
        if getattr(config, name) % tensor_parallel_size:
            raise ValueError(f"{name} must be divisible by tensor_parallel_size")
    if config.num_attention_heads % config.num_key_value_heads:
        raise ValueError("query heads must be divisible by KV heads")
    dtype = getattr(config, "dtype", None) or getattr(config, "torch_dtype", None)
    if str(dtype).removeprefix("torch.") not in ("float16", "bfloat16"):
        raise ValueError("checkpoint dtype must be float16 or bfloat16")


def build_model(config):
    # These families share checkpoint naming and RMSNorm + RoPE + GQA + SwiGLU.
    # QKV bias and QK normalization are independent switches.
    validate_model_config(config)
    from nanovllm.models.qwen3 import Qwen3ForCausalLM
    return Qwen3ForCausalLM(config)
