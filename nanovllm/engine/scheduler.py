from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.scheduling_policy = config.scheduling_policy
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        required = (len(seq) + seq.max_tokens - 1 + self.block_size - 1) // self.block_size
        if required > len(self.block_manager.blocks):
            raise ValueError("prompt cannot fit in the KV pool; shorten it or increase GPU memory")
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        if self.scheduling_policy == "decode_first":
            return self._schedule_decode_first()
        return self._schedule_prefill_first()

    def _schedule_decode_first(self) -> tuple[list[Sequence], bool]:
        """Reserve one token per decode, then fill the budget with prompt chunks.

        Mixed batches use the paged variable-length attention path; decode-only
        batches retain the existing CUDA graph path. KV for each prompt is still
        reserved in full, so this is token chunking, not incremental KV allocation.
        """
        scheduled = []
        budget = self.max_num_batched_tokens
        while self.running and budget and len(scheduled) < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                self.block_manager.may_append(seq)
                seq.is_prefill = False
                seq.num_scheduled_tokens = 1
                scheduled.append(seq)
                budget -= 1
        self.running.extendleft(reversed(scheduled))

        while self.waiting and budget and len(scheduled) < self.max_num_seqs:
            seq = self.waiting[0]
            if not seq.block_table:
                cached_blocks = self.block_manager.can_allocate(seq)
                if cached_blocks == -1:
                    break
                self.block_manager.allocate(seq, cached_blocks)
            seq.is_prefill = True
            seq.num_scheduled_tokens = min(len(seq) - seq.num_cached_tokens, budget)
            scheduled.append(seq)
            budget -= seq.num_scheduled_tokens
            if seq.num_cached_tokens + seq.num_scheduled_tokens < len(seq):
                break  # The head request keeps its place until its last chunk.
            self.waiting.popleft()
            seq.status = SequenceStatus.RUNNING
            self.running.append(seq)

        if not scheduled:
            # A decode can be preempted and become the only waiting request.
            # Retry admission after that preemption, or report a real capacity error.
            raise RuntimeError("no schedulable tokens; KV pool cannot admit the waiting request")
        return scheduled, any(seq.is_prefill for seq in scheduled)

    def _schedule_prefill_first(self) -> tuple[list[Sequence], bool]:
        scheduled_seqs = []
        num_batched_tokens = 0

        # prefill
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.waiting[0]
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining == 0:
                break
            if not seq.block_table:
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    break
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            if remaining < num_tokens and scheduled_seqs:  # only allow chunked prefill for the first seq
                break
            if not seq.block_table:
                self.block_manager.allocate(seq, num_cached_blocks)
            seq.num_scheduled_tokens = min(num_tokens, remaining)
            num_batched_tokens += seq.num_scheduled_tokens
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            scheduled_seqs.append(seq)

        if scheduled_seqs:
            return scheduled_seqs, True

        # decode
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            if len(scheduled_seqs) == self.max_num_batched_tokens:
                break
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        if not scheduled_seqs:
            if self.waiting and self.block_manager.can_allocate(self.waiting[0]) != -1:
                return self._schedule_prefill_first()
            raise RuntimeError("no schedulable tokens")
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.block_manager.deallocate(seq)
        # A partially computed prompt already owns KV blocks. Let it complete
        # before a preempted request tries to reserve a new full prompt.
        if self.waiting and self.waiting[0].block_table:
            self.waiting.insert(1, seq)
        else:
            self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int | None], is_prefill: bool):
        for seq, token_id in zip(seqs, token_ids):
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            if seq.num_cached_tokens < seq.num_tokens:
                assert token_id is None
                continue
            assert token_id is not None
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
