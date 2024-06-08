# vLLM部署
vLLM 是一个快速且易于使用的库，用于大型语言模型（LLM）的推理和服务。

vLLM 的速度优势包括：

- 最先进的服务吞吐量
- 使用 PagedAttention 高效管理注意力键和值的内存
- 连续批处理传入请求
- 使用 CUDA/HIP 图实现快速模型执行
- 量化支持：GPTQ、AWQ、SqueezeLLM、FP8 KV 缓存
- 优化的 CUDA 内核

vLLM 的灵活性和易用性包括：

- 与流行的 Hugging Face 模型无缝集成
- 通过各种解码算法（包括并行采样、束搜索等）实现高吞吐量服务
- 支持张量并行以进行分布式推理
- 流式输出
- 兼容 OpenAI 的 API 服务器
- 支持 NVIDIA GPU 和 AMD GPU
- （实验性）前缀缓存支持
- （实验性）多 lora 支持

vLLM 无缝支持 HuggingFace 上大多数流行的开源模型，包括：

- 类 Transformer 的大型语言模型（例如 Llama）
- 专家混合大型语言模型（例如 Mixtral）
- 多模态大型语言模型（例如 LLaVA）

## 简单推理
```python
from vllm import LLM, SamplingParams

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
llm = LLM(model="PATH-TO-YOUR-MODEL/MODEL-ID")

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```
