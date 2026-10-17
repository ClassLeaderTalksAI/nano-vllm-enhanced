import random
from types import SimpleNamespace
import pytest
from nanovllm import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler


@pytest.fixture(autouse=True)
def small_blocks(monkeypatch):
    monkeypatch.setattr(Sequence, "block_size", 4)


def scheduler(budget=3, blocks=32, policy="decode_first", max_seqs=8):
    return Scheduler(SimpleNamespace(max_num_seqs=max_seqs, max_num_batched_tokens=budget,
        eos=99, kvcache_block_size=4, num_kvcache_blocks=blocks, scheduling_policy=policy))


def seq(tokens, max_tokens=3, ignore_eos=True):
    return Sequence(tokens, SamplingParams(max_tokens=max_tokens, ignore_eos=ignore_eos))


def step(s, token=42):
    batch, prefill = s.schedule()
    assert 0 < sum(q.num_scheduled_tokens for q in batch) <= s.max_num_batched_tokens
    assert len(batch) <= s.max_num_seqs
    assert len({q.seq_id for q in batch}) == len(batch)
    out = [token if q.num_cached_tokens + q.num_scheduled_tokens == len(q) else None for q in batch]
    s.postprocess(batch, out, prefill)
    return batch, prefill


@pytest.mark.parametrize("policy", ["decode_first", "prefill_first"])
def test_chunks_do_not_emit_until_final_chunk(policy):
    s = scheduler(policy=policy)
    q = seq(list(range(8)))
    s.add(q)
    step(s)
    assert (q.num_cached_tokens, q.num_completion_tokens) == (3, 0)
    assert not s.block_manager.hash_to_block_id  # partial block isn't published
    step(s)
    assert (q.num_cached_tokens, q.num_completion_tokens) == (6, 0)
    assert len(s.block_manager.hash_to_block_id) == 1
    step(s)
    assert q.num_completion_tokens == 1
    while not s.is_finished():
        step(s)
    assert len(s.block_manager.free_block_ids) == 32


def test_decode_is_mixed_with_long_prompt_and_not_stalled():
    s = scheduler(budget=4)
    short = seq([1, 2], max_tokens=5)
    s.add(short)
    step(s)
    long = seq(list(range(15)))
    s.add(long)
    batch, prefill = step(s)
    assert batch == [short, long] and prefill
    assert short.num_completion_tokens == 2
    assert long.num_cached_tokens == 3 and long.num_completion_tokens == 0
    for _ in range(2):
        step(s)
    assert short.num_completion_tokens == 4


def test_baseline_prefill_priority_stalls_decode():
    s = scheduler(budget=4, policy="prefill_first")
    short = seq([1, 2], max_tokens=5)
    s.add(short)
    step(s)
    s.add(seq(list(range(15))))
    batch, prefill = step(s)
    assert batch != [short] and short not in batch and prefill
    assert short.num_completion_tokens == 1


def test_prefix_reuse_only_after_computation_and_leave_last_block():
    s = scheduler(budget=4)
    first = seq(list(range(8)), max_tokens=1)
    s.add(first)
    while not s.is_finished():
        step(s)
    second = seq(list(range(8)), max_tokens=1)
    assert s.block_manager.can_allocate(second) == 1
    s.add(second)
    batch, _ = s.schedule()
    assert second.num_cached_tokens == 4 and second.num_scheduled_tokens == 4
    s.postprocess(batch, [42], True)
    assert second.is_finished and not s.block_manager.used_block_ids


def test_eos_reclaims_all_blocks():
    s = scheduler(budget=20)
    q = seq([1, 2, 3], ignore_eos=False)
    s.add(q)
    step(s, 99)
    assert s.is_finished() and q.is_finished and len(s.block_manager.free_block_ids) == 32


@pytest.mark.parametrize("policy", ["decode_first", "prefill_first"])
def test_preemption_recompute_and_reclaim(policy):
    s = scheduler(budget=4, blocks=4, policy=policy)
    queries = [seq([1, 2, 3, 4], max_tokens=8), seq([6, 7, 8, 9], max_tokens=8)]
    for q in queries:
        s.add(q)
    for _ in range(200):
        if s.is_finished():
            break
        step(s)
    assert s.is_finished()
    assert all(q.num_completion_tokens == 8 for q in queries)
    assert len(s.block_manager.free_block_ids) == 4


def test_preempted_decode_does_not_block_resident_prompt():
    s = scheduler(budget=2, blocks=3)
    a = seq([1, 2, 3], max_tokens=8)
    s.add(a)
    step(s); step(s)
    b = seq(list(range(8)), max_tokens=1)
    s.add(b)
    step(s)  # a fills its block; b reserves two blocks and computes one token.
    assert b.num_cached_tokens == 1
    step(s)  # a needs a new block and is preempted behind resident b.
    assert s.waiting[0] is b and b.num_cached_tokens == 3
    for _ in range(100):
        if s.is_finished():
            break
        step(s)
    assert s.is_finished() and a.num_completion_tokens == 8
    assert len(s.block_manager.free_block_ids) == 3


@pytest.mark.parametrize("policy", ["decode_first", "prefill_first"])
def test_budget_bounds_and_resource_invariants_under_arrivals(policy):
    rng = random.Random(13)
    s = scheduler(budget=3, blocks=12, policy=policy, max_seqs=4)
    queries = []
    for tick in range(1000):
        if tick < 30 and rng.random() < .7:
            q = seq([rng.randrange(8) for _ in range(rng.randrange(1, 14))], max_tokens=rng.randrange(1, 7))
            queries.append(q)
            s.add(q)
        if not s.is_finished():
            step(s)
        bm = s.block_manager
        assert set(bm.free_block_ids).isdisjoint(bm.used_block_ids)
        assert len(bm.free_block_ids) + len(bm.used_block_ids) == 12
        residents = list(s.running) + list(s.waiting)
        for block in bm.blocks:
            assert block.ref_count == sum(q.block_table.count(block.block_id) for q in residents)
        if tick >= 30 and s.is_finished():
            break
    assert s.is_finished()
    assert all(q.num_completion_tokens == q.max_tokens for q in queries)


def test_impossible_prompt_fails_on_admission():
    with pytest.raises(ValueError, match="KV pool"):
        scheduler(blocks=1).add(seq([1]*5))
