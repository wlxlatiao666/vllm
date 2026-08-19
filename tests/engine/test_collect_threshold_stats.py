from vllm import LLM, SamplingParams

model_path = "/inspire/hdd/global_public/public_models/Qwen/Qwen2.5-7B-Instruct"
llm = LLM(
    model=model_path,
    dtype="float16",
    tensor_parallel_size=1,
    gpu_memory_utilization=0.8,
    enforce_eager=True
    # enable_chunked_prefill=True
)
prompt = "What is the meaning of life?"
params = SamplingParams(
    n=5,
    temperature=0.8,
    max_tokens=200,
    collect_threshold_stats=True,
    collect_importance_stats=True,
)
outputs = llm.generate([prompt], params)

for i in range(5):
    result = outputs[0].outputs[i]
    print('entropy_list:', result.entropy_list)      # 每个 output token 的 entropy
    print('importance_list:', result.importance_list)   # 每个 output token 的 waad importance（需要 attention 支持）
