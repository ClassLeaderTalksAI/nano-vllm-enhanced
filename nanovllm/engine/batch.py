"""Build packed attention inputs without depending on a CUDA runtime."""
from dataclasses import dataclass
from nanovllm.engine.sequence import Sequence


@dataclass
class PrefillInputs:
    input_ids: list[int]
    positions: list[int]
    cu_seqlens_q: list[int]
    cu_seqlens_k: list[int]
    slot_mapping: list[int]
    block_tables: list[list[int]] | None

    @property
    def max_seqlen_q(self):
        return max(b - a for a, b in zip(self.cu_seqlens_q, self.cu_seqlens_q[1:]))

    @property
    def max_seqlen_k(self):
        return max(b - a for a, b in zip(self.cu_seqlens_k, self.cu_seqlens_k[1:]))


def build_prefill_inputs(seqs: list[Sequence], block_size: int) -> PrefillInputs:
    data = PrefillInputs([], [], [0], [0], [], None)
    for seq in seqs:
        start = seq.num_cached_tokens
        end = start + seq.num_scheduled_tokens
        assert start < end <= len(seq)
        # Decode workers receive only last_token, even in a mixed batch.
        data.input_ids.extend(seq[start:end] if seq.is_prefill else [seq.last_token])
        data.positions.extend(range(start, end))
        data.cu_seqlens_q.append(data.cu_seqlens_q[-1] + end - start)
        data.cu_seqlens_k.append(data.cu_seqlens_k[-1] + end)
        if seq.block_table:  # Empty only during model warmup.
            data.slot_mapping.extend(seq.block_table[pos // block_size] * block_size + pos % block_size
                                     for pos in range(start, end))
    if data.cu_seqlens_k[-1] > data.cu_seqlens_q[-1]:
        width = max(len(seq.block_table) for seq in seqs)
        data.block_tables = [seq.block_table + [-1] * (width - len(seq.block_table)) for seq in seqs]
    return data
