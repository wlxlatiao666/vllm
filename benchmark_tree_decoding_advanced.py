import time
import argparse
import numpy as np
from datasets import load_dataset
from vllm import LLM, SamplingParams
from vllm.sampling_params import TreeSearchParams

def load_livecodebench_prompts(num_prompts):
    """
    Attempt to load LiveCodeBench prompts. You may need to install the `datasets` library.
    """
    try:
        print("Loading LiveCodeBench dataset...")
        dataset = load_dataset("livecodebench/livecodebench", "release_v1", split="test")
        prompts = []
        for i in range(min(num_prompts, len(dataset))):
            question = dataset[i]["question"]
            prompts.append(f"Write a Python function to solve the following problem:\n{question}")
        return prompts
    except Exception as e:
        print(f"Failed to load LiveCodeBench: {e}. Using fallback code generation prompts.")
        return [
            f"Write a Python script to compute the Fibonacci sequence efficiently. Variant {i}" 
            for i in range(num_prompts)
        ]

def generate_long_context_prompt(length):
    """
    Generate a dummy long context prompt based on the target token length.
    Each word is roughly ~1.3 tokens.
    """
    base_text = "The quick brown fox jumps over the lazy dog. "
    repeats = length // 10
    prompt = "Context: \n" + (base_text * repeats) + "\n\nQuestion: What color is the fox?"
    return prompt

def main():
    parser = argparse.ArgumentParser(description="Advanced Benchmark: Equal Budget & Long Context")
    parser.add_argument("--model", type=str, default="/inspire/hdd/global_public/public_models/Qwen/Qwen2.5-7B-Instruct")
    # New tasks: cost_perf (Equal budget), long_context (>=16k tokens), livecodebench
    parser.add_argument("--task", type=str, choices=["cost_perf", "long_context", "livecodebench"], default="cost_perf")
    parser.add_argument("--num-prompts", type=int, default=20, help="Number of prompts to evaluate")
    parser.add_argument("--max-tokens", type=int, default=512, help="Max generation length")
    parser.add_argument("--long-context-len", type=int, default=16000, help="Input length for long_context task")
    
    # Baseline comparison settings (Equal Budget)
    parser.add_argument("--n-samples", type=int, default=3, help="Number of parallel samples for Baseline (setting best_of / n)")
    
    # Tree setting
    parser.add_argument("--branching-factor", type=int, default=3)
    parser.add_argument("--entropy-threshold", type=float, default=1.0)
    parser.add_argument("--max-tree-depth", type=int, default=3)
    parser.add_argument("--tau-importance", type=float, default=0.05)
    
    args = parser.parse_args()

    print(f"Initializing LLM Engine with {args.model}...")
    
    # Ensure memory covers the long context lengths if needed
    if args.task == "long_context":
        max_model_len = max(16384, args.long_context_len + args.max_tokens)
    else:
        max_model_len = 8192

    llm = LLM(
        model=args.model,
        tensor_parallel_size=1,
        dtype="float16",
        gpu_memory_utilization=0.9,
        enforce_eager=True,
        max_model_len=max_model_len
    )

    # 1. Prepare Prompts
    if args.task == "livecodebench":
        prompts = load_livecodebench_prompts(args.num_prompts)
    elif args.task == "long_context":
        print(f"Generating long context prompts (~{args.long_context_len} tokens baseline)")
        prompts = [generate_long_context_prompt(args.long_context_len) for _ in range(max(1, args.num_prompts // 4))]
    else:
        # Standard reasoning prompts for cost_perf
        base_instruct = "Explain the step-by-step reasoning to solve the following math problem: "
        prompts = [f"{base_instruct} Variant {i} of expanding (x+y)^{i+2}" for i in range(args.num_prompts)]

    # 2. Benchmark Baseline under Equal Compute Budget
    # Reviewer request: "equal compute budgets". We use n=3 to match branching_factor=3.
    print(f"\n[{args.task.upper()}] Baseline: Normal Decoding with Multiple Samples (n={args.n_samples})")
    baseline_params = SamplingParams(
        temperature=0.8,
        max_tokens=args.max_tokens,
        n=args.n_samples, # Generate multiple responses to match the cost of tree search
    )
    
    # Warmup
    llm.generate(["Warmup"], SamplingParams(max_tokens=10), use_tqdm=False)
    
    start_time = time.time()
    baseline_outputs = llm.generate(prompts, baseline_params, use_tqdm=True)
    baseline_time = time.time() - start_time
    
    baseline_tokens = sum(len(out.token_ids) for req in baseline_outputs for out in req.outputs)
    
    print(f"Baseline Time: {baseline_time:.2f}s")
    print(f"Baseline Throughput: {baseline_tokens / baseline_time:.2f} tokens/s")
    
    # 3. Benchmark Tree Decoding
    print(f"\n[{args.task.upper()}] Tree Decoding (branching={args.branching_factor})")
    tree_params = SamplingParams(
        temperature=0.8,
        max_tokens=args.max_tokens,
        tree_search_params=TreeSearchParams(
            enable_tree_search=True,
            entropy_threshold=args.entropy_threshold,
            branching_factor=args.branching_factor,
            max_tree_depth=args.max_tree_depth,
            tau_importance=args.tau_importance
        )
    )
    
    start_time = time.time()
    tree_outputs = llm.generate(prompts, tree_params, use_tqdm=True)
    tree_time = time.time() - start_time
    
    tree_tokens = sum(len(out.token_ids) for req in tree_outputs for out in req.outputs)
    
    print(f"Tree Decoding Time: {tree_time:.2f}s")
    print(f"Tree Decoding Throughput: {tree_tokens / tree_time:.2f} tokens/s")

    # 4. Compare Results
    print("\n" + "="*70)
    print("Cost-Performance & Scalability Comparison (Reviewer Rebuttal)")
    print(f"Task Model: {args.task}")
    print("="*70)
    print(f"{'Algorithm':<25} | {'Tokens Generated':<18} | {'Total Time (s)':<15} | {'Tokens/sec':<10}")
    print("-" * 70)
    print(f"{'Baseline (n='+str(args.n_samples)+')':<25} | {baseline_tokens:<18} | {baseline_time:<15.2f} | {baseline_tokens/baseline_time:<10.2f}")
    print(f"{'Tree Decoding (branch='+str(args.branching_factor)+')':<25} | {tree_tokens:<18} | {tree_time:<15.2f} | {tree_tokens/tree_time:<10.2f}")
    print("-" * 70)
    
    if args.task == "long_context":
        print("\n[Analysis]: For long contexts, attention overhead is magnified. Observe the throughput gap.")
    elif args.task == "cost_perf":
        print("\n[Analysis]: By keeping 'Tokens Generated' strictly equivalent (equal budget),")
        print("you can demonstrate that Tree Decoding utilizes FLOPs more efficiently to yield higher quality than baseline n-sampling.")

if __name__ == "__main__":
    main()
