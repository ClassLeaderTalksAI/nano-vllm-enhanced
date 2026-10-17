import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    scheduling_policy: str = "decode_first"

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        if self.max_num_batched_tokens <= 0 or self.max_num_seqs <= 0 or self.max_model_len <= 0:
            raise ValueError("token budget, sequence limit and model length must be positive")
        if not 0 < self.gpu_memory_utilization < 1:
            raise ValueError("gpu_memory_utilization must be in (0, 1)")
        if self.scheduling_policy not in ("decode_first", "prefill_first"):
            raise ValueError("scheduling_policy must be decode_first or prefill_first")
        from nanovllm.models.registry import validate_model_config
        validate_model_config(self.hf_config, self.tensor_parallel_size)
