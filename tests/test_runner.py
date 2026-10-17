from unittest.mock import Mock
import pytest
import torch
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.engine.sequence import Sequence
from nanovllm.utils.context import set_context, get_context


def runner_for(logits):
    runner = ModelRunner.__new__(ModelRunner)
    runner.rank = 0
    runner.prepare_prefill = Mock(return_value=(None, None))
    runner.prepare_decode = Mock(return_value=(None, None))
    runner.run_model = Mock(return_value=logits)
    runner.prepare_sample = Mock(return_value=(torch.ones(1), None, None))
    runner.sampler = Mock(return_value=torch.tensor([7]))
    return runner


def test_no_sampler_for_intermediate_chunks():
    runner = runner_for(torch.ones(1, 8))
    query = Sequence([1, 2, 3, 4])
    query.num_scheduled_tokens = 2
    set_context(True)
    assert runner.run([query], True) == [None]
    runner.sampler.assert_not_called()
    runner.prepare_sample.assert_not_called()
    assert not get_context().is_prefill


def test_mixed_run_only_samples_ready_rows():
    logits = torch.arange(16).view(2, 8)
    runner = runner_for(logits)
    intermediate = Sequence([1, 2, 3, 4])
    intermediate.num_scheduled_tokens = 2
    decode = Sequence([5, 6, 7])
    decode.is_prefill = False
    decode.num_cached_tokens = 2
    decode.num_scheduled_tokens = 1
    assert runner.run([intermediate, decode], True) == [None, 7]
    runner.prepare_sample.assert_called_once_with([decode])
    torch.testing.assert_close(runner.sampler.call_args.args[0], logits[1:])


def test_context_cleared_on_failed_forward():
    runner = runner_for(None)
    runner.run_model.side_effect = RuntimeError("forward failed")
    set_context(True)
    with pytest.raises(RuntimeError, match="forward failed"):
        runner.run([Sequence([1])], True)
    assert not get_context().is_prefill
