from llama_cpp import Llama
llm = Llama(
    model_path="models/mistral-7b-instruct-v0.3.Q4_K.gguf",
    n_ctx=4096,
    n_threads=6,           # or os.cpu_count()
    n_gpu_layers=-1        # load as many layers as fit GPU RAM
)
prompt = "### Instruction: Explain why methane and oxygen together can indicate life.\n### Response:"
out = llm(prompt, max_tokens=120, stop=["###"] )
print(out["choices"][0]["text"].strip())