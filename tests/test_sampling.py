import pickle
import pytest
import torch
from nanovllm import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.layers.sampler import Sampler, filter_logits


@pytest.mark.parametrize("kwargs", [dict(top_k=-1), dict(top_k=1.5), dict(top_k=True),
    dict(top_p=0), dict(top_p=1.01), dict(top_p=float("nan")), dict(temperature=-1),
    dict(temperature=float("inf")), dict(max_tokens=0)])
def test_invalid_params(kwargs):
    with pytest.raises(ValueError):
        SamplingParams(**kwargs)


def test_top_k_then_top_p_and_per_row_controls():
    logits = torch.tensor([[.4, .3, .2, .1]]).log().repeat(4, 1)
    filtered = filter_logits(logits, torch.tensor([0, 2, 0, 100]), torch.tensor([.6, .6, .01, 1.]))
    # Row 1 renormalizes top-2 probabilities to 4/7, 3/7, so both remain.
    assert torch.isfinite(filtered).tolist() == [[True, True, False, False],
        [True, True, False, False], [True, False, False, False], [True]*4]
    torch.testing.assert_close(logits[0], torch.tensor([.4, .3, .2, .1]).log())


def test_top_k_has_exact_cardinality_with_ties():
    out = filter_logits(torch.zeros(2, 5), torch.tensor([1, 3]), torch.ones(2))
    assert torch.isfinite(out).sum(1).tolist() == [1, 3]


def test_mixed_greedy_and_random_sampling():
    torch.manual_seed(7)
    logits = torch.tensor([[0., 2., 1.]]).repeat(3, 1)
    for _ in range(20):
        ids = Sampler()(logits, torch.tensor([0., .7, 1.]),
                        torch.tensor([0, 1, 0]), torch.tensor([1., 1., .01]))
        assert ids.tolist() == [1, 1, 1]


def test_unfiltered_sampling_distribution():
    torch.manual_seed(9)
    probs = torch.tensor([.1, .3, .6])
    n = 20000
    ids = Sampler()(probs.log().repeat(n, 1), torch.ones(n))
    observed = ids.bincount(minlength=3) / n
    torch.testing.assert_close(observed, probs, atol=.015, rtol=0)


@pytest.mark.parametrize("prefill", [True, False])
def test_worker_serialization(prefill):
    seq = Sequence([1, 2, 3], SamplingParams(temperature=.6, top_k=12, top_p=.8))
    seq.is_prefill = prefill
    seq.num_cached_tokens = 2
    seq.num_scheduled_tokens = 1
    seq.block_table = [4]
    worker = pickle.loads(pickle.dumps(seq))
    assert (worker.is_prefill, worker.last_token, worker.temperature, worker.top_k, worker.top_p) == (prefill, 3, .6, 12, .8)
    assert worker.block_table == [4]
    assert worker.token_ids == ([1, 2, 3] if prefill else [])
