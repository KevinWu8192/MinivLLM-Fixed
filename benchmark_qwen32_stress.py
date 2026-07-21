"""Qwen3-32B two-GPU stress tests for cache reuse and generation throughput.

The benchmark provides four independent suites:

1. prefix-cache: long shared system/multi-turn prefixes, cold versus hot;
2. high-concurrency: many concurrent requests and aggregate throughput;
3. long-decode: sustained decoding with multi-thousand-token completions;
4. max-output: the 8K prompt + 32K output boundary on one request.

Example:
    HF_HUB_DISABLE_XET=1 uv run python benchmark_qwen32_stress.py \
        --model Qwen/Qwen3-32B --suite prefix-cache
"""

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent / "src"))

from main_qwen32 import config as qwen32_config
from myvllm.engine.llm_engine import LLMEngine
from myvllm.sampling_parameters import SamplingParams


OFFICIAL_MAX_POSITION = 40_960
NATIVE_CONTEXT_LENGTH = 32_768


@dataclass
class BenchmarkResult:
    name: str
    requests: int
    prompt_tokens: int
    computed_prefill_tokens: int
    cached_prompt_tokens: int
    cache_hit_rate: float
    output_tokens: int
    prefill_seconds: float
    decode_seconds: float
    total_seconds: float
    prefill_tokens_per_second: float
    decode_tokens_per_second: float
    output_tokens_per_second: float
    requests_per_second: float


@dataclass(frozen=True)
class SuiteSpec:
    name: str
    description: str
    concurrency: int
    prompt_tokens: int
    output_tokens: int
    measures_cache: bool = False


SUITES = {
    "prefix-cache": SuiteSpec(
        name="prefix-cache",
        description="Long shared prefix: compare cold and prefix-cache-hit runs",
        concurrency=8,
        prompt_tokens=16_384,
        output_tokens=256,
        measures_cache=True,
    ),
    "high-concurrency": SuiteSpec(
        name="high-concurrency",
        description="High request concurrency and aggregate token throughput",
        concurrency=16,
        prompt_tokens=4_096,
        output_tokens=256,
    ),
    "long-decode": SuiteSpec(
        name="long-decode",
        description="Sustained decode throughput with long completions",
        concurrency=2,
        prompt_tokens=8_192,
        output_tokens=4_096,
    ),
    "max-output": SuiteSpec(
        name="max-output",
        description="Single-request 8K prompt plus 32K output boundary",
        concurrency=1,
        prompt_tokens=8_192,
        output_tokens=32_768,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stress Qwen3-32B TP=2 with long-context cache workloads.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Default suite shapes:
  prefix-cache      concurrency=8,  prompt=16384, output=256
  high-concurrency  concurrency=16, prompt=4096,  output=256
  long-decode       concurrency=2,  prompt=8192,  output=4096
  max-output        concurrency=1,  prompt=8192,  output=32768
""",
    )
    parser.add_argument("--model", default="Qwen/Qwen3-32B")
    parser.add_argument(
        "--suite",
        choices=[*SUITES, "all"],
        default="prefix-cache",
        help=(
            "Independent test suite to run. 'all' reloads the model for each "
            "suite so every test gets an appropriate engine shape."
        ),
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        help="Override the selected suite's request concurrency.",
    )
    parser.add_argument(
        "--prompt-tokens",
        type=int,
        help="Override the selected suite's approximate prompt length.",
    )
    parser.add_argument(
        "--output-tokens",
        type=int,
        help="Override the selected suite's output length per request.",
    )
    parser.add_argument("--seed-output-tokens", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Disable CUDA Graphs; useful for the first correctness run.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        help="Optionally write the machine-readable result to this path.",
    )
    return parser.parse_args()


def render_chat(tokenizer, messages: list[dict[str, str]]) -> str:
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def conversation(system_prompt: str, final_question: str) -> list[dict[str, str]]:
    """Create a shared system prompt plus deterministic multi-turn history."""
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": "State the operating rules you will follow in one sentence.",
        },
        {
            "role": "assistant",
            "content": (
                "I will use the supplied reference, preserve factual details, "
                "and answer concisely."
            ),
        },
        {
            "role": "user",
            "content": "Remember that every final answer must include a short conclusion.",
        },
        {
            "role": "assistant",
            "content": "Understood; every final answer will end with a short conclusion.",
        },
        {"role": "user", "content": final_question},
    ]


def build_long_system_prompt(tokenizer, target_prompt_tokens: int) -> str:
    """Find the largest repeated system prompt that stays under the target."""
    unit = (
        "Reference segment: distributed inference uses tensor parallel linear "
        "layers, paged KV cache blocks, prefix hashing, continuous scheduling, "
        "Flash Attention during Prefill, and paged Attention during Decode. "
        "Treat this segment as immutable context for the benchmark.\n"
    )
    final_question = "Give the request-specific benchmark conclusion."

    def prompt_length(repetitions: int) -> int:
        text = render_chat(
            tokenizer,
            conversation(unit * repetitions, final_question),
        )
        return len(tokenizer.encode(text))

    low, high = 0, 1
    while prompt_length(high) <= target_prompt_tokens:
        low, high = high, high * 2

    while low + 1 < high:
        middle = (low + high) // 2
        if prompt_length(middle) <= target_prompt_tokens:
            low = middle
        else:
            high = middle

    return unit * low


def make_workloads(tokenizer, concurrency: int, target_tokens: int):
    # Leave room for request-specific text while retaining a long shared body.
    shared_system = build_long_system_prompt(tokenizer, target_tokens - 32)

    cold_prompts = [
        render_chat(
            tokenizer,
            conversation(
                f"Cold request {index:04d}; this marker makes block zero unique.\n"
                + shared_system,
                f"Analyze cold workload request {index:04d} and give its conclusion.",
            ),
        )
        for index in range(concurrency)
    ]
    seed_prompt = render_chat(
        tokenizer,
        conversation(shared_system, "Seed the shared prefix cache now."),
    )
    hot_prompts = [
        render_chat(
            tokenizer,
            conversation(
                shared_system,
                f"Analyze hot workload request {index:04d} and give its conclusion.",
            ),
        )
        for index in range(concurrency)
    ]
    return cold_prompts, seed_prompt, hot_prompts


def synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def safe_rate(tokens_or_requests: int, seconds: float) -> float:
    return tokens_or_requests / seconds if seconds > 0 else 0.0


def run_workload(
    engine: LLMEngine,
    name: str,
    prompts: list[str],
    output_tokens: int,
) -> BenchmarkResult:
    prompt_tokens = sum(len(engine.tokenizer.encode(prompt)) for prompt in prompts)
    sampling = SamplingParams(
        temperature=0.6,
        max_tokens=output_tokens,
        ignore_eos=True,
        max_model_length=engine.config["max_model_length"],
    )
    for prompt in prompts:
        engine.add_prompt(prompt, sampling)

    computed_prefill_tokens = 0
    decoded_tokens = 0
    prefill_seconds = 0.0
    decode_seconds = 0.0
    completed: dict[int, list[int]] = {}

    synchronize()
    workload_start = time.perf_counter()
    while not engine.scheduler.is_finished():
        synchronize()
        step_start = time.perf_counter()
        outputs, processed_tokens, is_prefill = engine.step()
        synchronize()
        elapsed = time.perf_counter() - step_start

        if processed_tokens == 0 and not outputs:
            raise RuntimeError(
                f"{name}: scheduler made no progress; reduce concurrency or "
                "prompt length because the KV cache cannot admit the workload"
            )

        if is_prefill:
            computed_prefill_tokens += processed_tokens
            prefill_seconds += elapsed
        else:
            decoded_tokens += processed_tokens
            decode_seconds += elapsed
        completed.update(dict(outputs))

    synchronize()
    total_seconds = time.perf_counter() - workload_start
    output_token_count = sum(len(tokens) for tokens in completed.values())
    cached_tokens = max(0, prompt_tokens - computed_prefill_tokens)

    if len(completed) != len(prompts):
        raise RuntimeError(
            f"{name}: completed {len(completed)} of {len(prompts)} requests"
        )
    expected_output_tokens = len(prompts) * output_tokens
    if output_token_count != expected_output_tokens:
        raise RuntimeError(
            f"{name}: generated {output_token_count} tokens; "
            f"expected {expected_output_tokens}"
        )

    return BenchmarkResult(
        name=name,
        requests=len(prompts),
        prompt_tokens=prompt_tokens,
        computed_prefill_tokens=computed_prefill_tokens,
        cached_prompt_tokens=cached_tokens,
        cache_hit_rate=cached_tokens / prompt_tokens if prompt_tokens else 0.0,
        output_tokens=output_token_count,
        prefill_seconds=prefill_seconds,
        decode_seconds=decode_seconds,
        total_seconds=total_seconds,
        prefill_tokens_per_second=safe_rate(
            computed_prefill_tokens, prefill_seconds
        ),
        decode_tokens_per_second=safe_rate(decoded_tokens, decode_seconds),
        output_tokens_per_second=safe_rate(output_token_count, total_seconds),
        requests_per_second=safe_rate(len(prompts), total_seconds),
    )


def print_result(result: BenchmarkResult) -> None:
    print(f"\n=== {result.name} ===")
    print(f"requests:                 {result.requests}")
    print(f"prompt tokens:             {result.prompt_tokens:,}")
    print(f"computed prefill tokens:   {result.computed_prefill_tokens:,}")
    print(f"cached prompt tokens:      {result.cached_prompt_tokens:,}")
    print(f"cache hit rate:            {result.cache_hit_rate:.2%}")
    print(f"output tokens:             {result.output_tokens:,}")
    print(f"prefill seconds:           {result.prefill_seconds:.3f}")
    print(f"decode seconds:            {result.decode_seconds:.3f}")
    print(f"total seconds:             {result.total_seconds:.3f}")
    print(
        f"computed prefill tok/s:    {result.prefill_tokens_per_second:,.2f}"
    )
    print(f"decode tok/s:              {result.decode_tokens_per_second:,.2f}")
    print(f"end-to-end output tok/s:   {result.output_tokens_per_second:,.2f}")
    print(f"requests/s:                {result.requests_per_second:.4f}")


def selected_suites(args: argparse.Namespace) -> list[SuiteSpec]:
    if args.suite == "all":
        if any(
            value is not None
            for value in (
                args.concurrency,
                args.prompt_tokens,
                args.output_tokens,
            )
        ):
            raise ValueError(
                "concurrency/prompt/output overrides require one specific suite"
            )
        return list(SUITES.values())

    base = SUITES[args.suite]
    return [
        SuiteSpec(
            name=base.name,
            description=base.description,
            concurrency=(
                base.concurrency
                if args.concurrency is None
                else args.concurrency
            ),
            prompt_tokens=(
                base.prompt_tokens
                if args.prompt_tokens is None
                else args.prompt_tokens
            ),
            output_tokens=(
                base.output_tokens
                if args.output_tokens is None
                else args.output_tokens
            ),
            measures_cache=base.measures_cache,
        )
    ]


def validate_suite(spec: SuiteSpec, seed_output_tokens: int) -> None:
    if spec.concurrency <= 0:
        raise ValueError("concurrency must be greater than 0")
    if spec.prompt_tokens <= 0 or spec.output_tokens <= 0:
        raise ValueError("prompt-tokens and output-tokens must be greater than 0")
    if seed_output_tokens <= 0:
        raise ValueError("seed-output-tokens must be greater than 0")
    if spec.prompt_tokens + spec.output_tokens > OFFICIAL_MAX_POSITION:
        raise ValueError(
            f"{spec.name}: prompt-tokens + output-tokens exceeds the official "
            f"Qwen3-32B capacity of {OFFICIAL_MAX_POSITION}"
        )
    if spec.prompt_tokens > NATIVE_CONTEXT_LENGTH:
        print(
            "WARNING: prompt length exceeds the documented 32,768-token "
            "native context. This engine does not implement YaRN."
        )


def run_suite(
    args: argparse.Namespace,
    tokenizer,
    spec: SuiteSpec,
) -> dict:
    validate_suite(spec, args.seed_output_tokens)
    cold_prompts, seed_prompt, hot_prompts = make_workloads(
        tokenizer, spec.concurrency, spec.prompt_tokens
    )
    all_stress_prompts = (
        cold_prompts + hot_prompts if spec.measures_cache else cold_prompts
    )
    actual_max_prompt = max(len(tokenizer.encode(p)) for p in all_stress_prompts)
    if actual_max_prompt + spec.output_tokens > OFFICIAL_MAX_POSITION:
        raise ValueError(
            f"Rendered prompt length {actual_max_prompt} plus output length "
            f"{spec.output_tokens} exceeds {OFFICIAL_MAX_POSITION}"
        )
    engine_max_model_length = actual_max_prompt + spec.output_tokens

    config = dict(qwen32_config)
    config.update(
        model_name_or_path=args.model,
        world_size=2,
        block_size=args.block_size,
        max_num_sequences=spec.concurrency,
        max_num_batched_tokens=sum(
            len(tokenizer.encode(prompt)) for prompt in cold_prompts
        ),
        max_position=OFFICIAL_MAX_POSITION,
        # Keep the engine limit at the requested stress shape. ModelRunner uses
        # this value for warmup and CUDA Graph Block Tables, while RoPE retains
        # the full official 40,960-position capacity.
        max_model_length=engine_max_model_length,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
    )

    print(f"\n{'=' * 72}")
    print(f"Suite: {spec.name} - {spec.description}")
    print("=== Qwen3-32B stress configuration ===")
    print(f"model:                     {args.model}")
    print("tensor parallel size:      2")
    print(f"concurrency:               {spec.concurrency}")
    print(f"target prompt tokens:      {spec.prompt_tokens:,}")
    print(f"actual max prompt tokens:  {actual_max_prompt:,}")
    print(f"output tokens/request:     {spec.output_tokens:,}")
    print(f"engine max model length:   {engine_max_model_length:,}")
    print(f"max batched tokens:        {config['max_num_batched_tokens']:,}")
    print(f"CUDA Graphs enabled:       {not args.enforce_eager}")
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            print(
                f"GPU {index}: {properties.name}, "
                f"{properties.total_memory / 2**30:.1f} GiB"
            )

    engine = LLMEngine(config=config)
    try:
        if spec.measures_cache:
            results = [
                run_workload(
                    engine,
                    "cold_unique",
                    cold_prompts,
                    spec.output_tokens,
                ),
                run_workload(
                    engine,
                    "cache_seed",
                    [seed_prompt],
                    args.seed_output_tokens,
                ),
                run_workload(
                    engine,
                    "hot_shared",
                    hot_prompts,
                    spec.output_tokens,
                ),
            ]
        else:
            results = [
                run_workload(
                    engine,
                    spec.name,
                    cold_prompts,
                    spec.output_tokens,
                )
            ]
    finally:
        engine.exit()

    for result in results:
        print_result(result)

    return {
        "configuration": {
            "suite": spec.name,
            "description": spec.description,
            "model": args.model,
            "tensor_parallel_size": 2,
            "concurrency": spec.concurrency,
            "target_prompt_tokens": spec.prompt_tokens,
            "actual_max_prompt_tokens": actual_max_prompt,
            "output_tokens": spec.output_tokens,
            "max_position": OFFICIAL_MAX_POSITION,
            "engine_max_model_length": engine_max_model_length,
            "native_context_length": NATIVE_CONTEXT_LENGTH,
            "enforce_eager": args.enforce_eager,
        },
        "results": [asdict(result) for result in results],
    }


def main() -> None:
    args = parse_args()
    specs = selected_suites(args)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    summaries = [run_suite(args, tokenizer, spec) for spec in specs]

    summary = {
        "selected_suite": args.suite,
        "suites": summaries,
    }
    if args.json_output:
        args.json_output.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
        )
        print(f"\nJSON result written to {args.json_output}")


if __name__ == "__main__":
    main()
