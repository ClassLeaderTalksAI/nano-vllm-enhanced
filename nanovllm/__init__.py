from nanovllm.sampling_params import SamplingParams

__all__ = ["LLM", "SamplingParams"]


def __getattr__(name):
    # Scheduling and sampling tests can run without the CUDA-only backend.
    if name == "LLM":
        from nanovllm.llm import LLM
        return LLM
    raise AttributeError(name)
