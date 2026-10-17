import pickle
from nanovllm.engine.batch import build_prefill_inputs
from nanovllm.engine.sequence import Sequence


def test_mixed_batch_has_exact_offsets_slots_and_worker_equivalence(monkeypatch):
    monkeypatch.setattr(Sequence, "block_size", 4)
    decode = Sequence([10, 11, 12, 13, 14])
    decode.is_prefill = False
    decode.num_cached_tokens = 4
    decode.num_scheduled_tokens = 1
    decode.block_table = [2, 5]
    prompt = Sequence([20, 21, 22, 23, 24, 25, 26])
    prompt.num_cached_tokens = 3
    prompt.num_scheduled_tokens = 3
    prompt.block_table = [7, 1]
    batch = build_prefill_inputs([decode, prompt], 4)
    assert batch.input_ids == [14, 23, 24, 25]
    assert batch.positions == [4, 3, 4, 5]
    assert batch.cu_seqlens_q == [0, 1, 4]
    assert batch.cu_seqlens_k == [0, 5, 11]
    assert batch.slot_mapping == [20, 31, 4, 5]
    assert batch.block_tables == [[2, 5], [7, 1]]
    assert (batch.max_seqlen_q, batch.max_seqlen_k) == (3, 6)
    workers = pickle.loads(pickle.dumps([decode, prompt]))
    assert build_prefill_inputs(workers, 4) == batch


def test_new_prompt_uses_contiguous_attention_and_padded_tables(monkeypatch):
    monkeypatch.setattr(Sequence, "block_size", 4)
    a, b = Sequence([1]*3), Sequence([2]*6)
    a.block_table, b.block_table = [4], [2, 6]
    a.num_scheduled_tokens, b.num_scheduled_tokens = 3, 6
    data = build_prefill_inputs([a, b], 4)
    assert data.block_tables is None
    assert data.slot_mapping == [16, 17, 18, 8, 9, 10, 11, 24, 25]
    a.num_cached_tokens, a.num_scheduled_tokens = 2, 1
    data = build_prefill_inputs([a, b], 4)
    assert data.block_tables == [[4, -1], [2, 6]]
