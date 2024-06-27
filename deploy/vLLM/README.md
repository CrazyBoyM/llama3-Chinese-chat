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

## 安装
```
pip install vllm
```
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
## OpenAI格式后端部署
```
python -m vllm.entrypoints.openai.api_server \
    --chat-template /path/to/your-chat-template.jinja \
    --model /path/to/your_model \
    --served-model-name llama3-cn \
    --max-model-len=1024
```
支持的全部参数列表如下，可按需自行调整：

| 参数                     | 说明                                                                                                  | 默认值          |
|-----------------------|-----------------------------------------------------------------------------------------------------|---------------|
| --host                | 主机名                                                                                                |               |
| --port                | 端口号                                                                                                | 8000          |
| --uvicorn-log-level   | uvicorn的日志级别，可选值为debug, info, warning, error, critical, trace                               | "info"        |
| --allow-credentials   | 是否允许凭证                                                                                          | False         |
| --allowed-origins     | 允许的来源                                                                                            | ['*']         |
| --allowed-methods     | 允许的方法                                                                                            | ['*']         |
| --allowed-headers     | 允许的头部                                                                                            | ['*']         |
| --api-key             | 如果提供，服务器将要求在标头中呈现此密钥。                                                                 |               |
| --lora-modules        | LoRA模块配置，格式为name=path。可以指定多个模块。                                                         |               |
| --chat-template       | 聊天模板的文件路径，或指定模型的单行形式模板。                                                            |               |
| --response-role       | 如果request.add_generation_prompt=true，则返回的角色名称。                                                  | "assistant"   |
| --ssl-keyfile         | SSL密钥文件的路径                                                                                       |               |
| --ssl-certfile        | SSL证书文件的路径                                                                                       |               |
| --ssl-ca-certs        | CA证书文件的路径                                                                                        |               |
| --ssl-cert-reqs       | 是否需要客户端证书                                                                                       | 0             |
| --root-path           | FastAPI的root_path，当应用程序在基于路径的路由代理后面时使用。                                                |               |
| --middleware          | 应用于应用程序的额外ASGI中间件。接受多个--middleware参数。值应为导入路径。                                    | []            |
| --model               | 要使用的huggingface模型的名称或路径。                                                                      | "facebook/opt-125m" |
| --tokenizer           | 要使用的huggingface分词器的名称或路径。如果未指定，则将使用模型名称或路径。                                    |               |
| --skip-tokenizer-init | 跳过分词器和解分词器的初始化。                                                                          | False         |
| --revision            | 要使用的特定模型版本。可以是分支名称、标签名称或提交ID。如果未指定，将使用默认版本。                               |               |
| --code-revision       | 在Hugging Face Hub上使用的模型代码的特定修订版。可以是分支名称、标签名称或提交ID。如果未指定，将使用默认版本。             |               |
| --tokenizer-revision  | 要使用的huggingface分词器的修订版本。可以是分支名称、标签名称或提交ID。如果未指定，将使用默认版本。                      |               |
| --tokenizer-mode     | 分词器模式，可选值为auto, slow。                                                                         | "auto"        |
| --trust-remote-code   | 是否信任来自Hugging Face的远程代码。                                                                    | False         |
| --download-dir        | 下载和加载权重的目录，默认为huggingface的默认缓存目录。                                                        |               |
| --load-format         | 加载模型权重的格式，可选值为auto, pt, safetensors, npcache, dummy, tensorizer, bitsandbytes。                     | "auto"        |
| --dtype               | 模型权重和激活的数据类型，可选值为auto, half, float16, bfloat16, float, float32。                                 | "auto"        |
| --kv-cache-dtype      | kv缓存存储的数据类型，可选值为auto, fp8, fp8_e5m2, fp8_e4m3。如果为"auto"，将使用模型的数据类型。                           | "auto"        |
| --quantization-param-path | 包含KV缓存缩放因子的JSON文件的路径。当KV缓存数据类型为FP8时，通常应提供此文件。否则，KV缓存缩放因子默认为1.0，可能导致精度问题。 |               |
| --max-model-len       | 模型上下文长度。如果未指定，将自动从模型配置中获取。                                                            |               |
| --guided-decoding-backend | 引导解码的引擎，可选值为outlines, lm-format-enforcer。默认为outlines。                                          | "outlines"    |
| --distributed-executor-backend | 分布式服务的后端，可选值为ray, mp。当使用多个GPU时，如果安装了Ray，则自动设置为"ray"，否则设置为"mp"（多进程）。     |               |
| --worker-use-ray      | 是否使用Ray启动LLM引擎作为单独的进程。已弃用，请使用--distributed-executor-backend=ray。                              | False         |
| --pipeline-parallel-size, -pp | 管道并行的阶段数。                                                                                   | 1             |
| --tensor-parallel-size, -tp | 张量并行的副本数。                                                                                   | 1             |
| --max-parallel-loading-workers | 在使用张量并行和大模型时，按顺序加载模型以避免RAM OOM。                                                 |               |
| --ray-workers-use-nsight | 如果指定，使用nsight对Ray工作进程进行性能分析。                                                       | False         |
| --block-size          | 连续的令牌块大小，可选值为8, 16, 32。                                                                      | 16            |
| --enable-prefix-caching | 启用自动前缀缓存。                                                                                   | False         |
| --disable-sliding-window | 禁用滑动窗口，限制为滑动窗口大小。                                                                         | False         |
| --use-v2-block-manager | 使用BlockSpaceMangerV2。                                                                              | False         |
| --num-lookahead-slots | 用于推测解码的实验性调度配置。在将来将被speculative config取代；目前用于启用正确性测试。                            | 0             |
| --seed                | 操作的随机种子。                                                                                      | 0             |
| --swap-space          | 每个GPU的CPU交换空间大小（以GiB为单位）。                                                                    | 4             |
| --gpu-memory-utilization | 模型执行器使用的GPU内存比例，范围从0到1。例如，0.5表示50%的GPU内存利用率。如果未指定，将使用默认值0.9。                     | 0.9           |
| --num-gpu-blocks-override | 如果指定，忽略GPU分析结果，并使用此GPU块数。用于测试抢占。                                                   |               |
| --max-num-batched-tokens | 每次迭代的最大批量令牌数。                                                                                |               |
| --max-num-seqs        | 每次迭代的最大序列数。                                                                                  | 256           |
| --max-logprobs        | 返回logprobs的最大数量，logprobs在SamplingParams中指定。                                                  | 20            |
| --disable-log-stats   | 禁用日志统计信息。                                                                                     | False         |
| --quantization, -q    | 权重量化的方法，可选值为aqlm, awq, deepspeedfp, fp8, marlin, gptq_marlin_24, gptq_marlin, gptq, squeezellm, compressed-tensors, bitsandbytes, None。 | |
| --rope-scaling        | RoPE缩放配置的JSON格式。例如，{"type":"dynamic","factor":2.0}。                                           |               |
| --rope-theta          | RoPE theta。与rope_scaling一起使用。在某些情况下，更改RoPE theta可以提高缩放模型的性能。                            |               |
| --enforce-eager       | 始终使用eager-mode PyTorch。如果为False，将使用eager模式和CUDA图形混合以实现最大性能和灵活性。                        | False         |
| --max-context-len-to-capture | CUDA图形捕获的最大上下文长度。当一个序列的上下文长度大于此值时，将回退到eager模式。                           |               |
| --max-seq-len-to-capture | CUDA图形捕获的最大序列长度。当一个序列的上下文长度大于此值时，将回退到eager模式。                             | 8192          |
| --disable-custom-all-reduce | 见ParallelConfig。                                                                                   | False         |
| --tokenizer-pool-size | 用于异步分词的分词器池的大小。如果为0，将使用同步分词。                                                        | 0             |
| --tokenizer-pool-type | 用于异步分词的分词器池的类型。如果tokenizer_pool_size为0，则忽略此参数。                                         | "ray"         |
| --tokenizer-pool-extra-config | 分词器池的额外配置。这应该是一个JSON字符串，将被解析为字典。如果tokenizer_pool_size为0，则忽略此参数。                  |               |
| --enable-lora         | 是否启用LoRA适配器的处理。                                                                             | False         |
| --max-loras           | 单个批次中的最大LoRA数量。                                                                               | 1             |
| --max-lora-rank       | 最大的LoRA rank。                                                                                      | 16            |
| --lora-extra-vocab-size | LoRA适配器中可以存在的额外词汇的最大大小（添加到基本模型词汇表中）。                                                 | 256           |
| --lora-dtype          | LoRA的数据类型，可选值为auto, float16, bfloat16, float32。如果为"auto"，将使用基本模型的数据类型。                        | "auto"        |
| --long-lora-scaling-factors | 指定多个缩放因子（可以不同于基本模型的缩放因子）以允许同时使用使用这些缩放因子训练的多个LoRA适配器。如果未指定，只允许使用使用基本模型缩放因子训练的适配器。 | |
| --max-cpu-loras       | 存储在CPU内存中的最大LoRA数量。必须大于等于max_num_seqs。默认为max_num_seqs。                                      |               |
| --fully-sharded-loras | 默认情况下，只有一半的LoRA计算使用张量并行。启用此选项将使用完全分片的层。在高序列长度、最大rank或张量并行大小下，这可能更快。         | False         |
| --device              | 设备类型，可选值为auto, cuda, neuron, cpu, tpu。                                                           | "auto"        |
| --image-input-type    | 传递给vLLM的图像输入类型，可选值为pixel_values, image_features。                                              |               |
| --image-token-id      | 图像令牌的输入ID。                                                                                     |               |
| --image-input-shape   | 给定输入类型的最大图像输入形状（内存占用最大）。仅用于vLLM的profile_run。                                       |               |
| --image-feature-size  | 图像特征在上下文维度上的大小。                                                                                |               |
| --image-processor     | 要使用的huggingface图像处理器的名称或路径。如果未指定，则将使用模型名称或路径。                                    |               |
| --image-processor-revision | 要使用的huggingface图像处理器版本的修订版本。可以是分支名称、标签名称或提交ID。如果未指定，将使用默认版本。                |               |
| --disable-image-processor | 禁用图像处理器的使用，即使在huggingface的模型中定义了图像处理器。                                                 | False         |
| --scheduler-delay-factor | 在调度下一个提示之前应用延迟（延迟因子乘以先前提示的延迟）。                                                   | 0.0           |
| --enable-chunked-prefill | 如果设置，预填充请求可以根据max_num_batched_tokens进行分块。                                                 | False         |
| --speculative-model   | 用于推测解码的草稿模型的名称。                                                                             |               |
| --num-speculative-tokens | 在推测解码中从草稿模型中采样的推测令牌数量。                                                                  |               |
| --speculative-max-model-len | 草稿模型支持的最大序列长度。超过此长度的序列将跳过推测解码。                                                   |               |
| --speculative-disable-by-batch-size | 如果新的入站请求的排队请求数大于该值，则禁用推测解码。                                                    |               |
| --ngram-prompt-lookup-max | 用于推测解码中ngram提示查找的窗口的最大大小。                                                                 |               |
| --ngram-prompt-lookup-min | 用于推测解码中ngram提示查找的窗口的最小大小。                                                                 |               |
| --model-loader-extra-config | 模型加载器的额外配置。将传递给相应的模型加载器。这应该是一个JSON字符串，将被解析为字典。                             |               |
| --preemption_mode     | 如果为'recompute'，则引擎通过块交换执行抢占；如果为'swap'，则引擎通过块交换执行抢占。                                |               |
| --served-model-name   | API中使用的模型名称。如果提供多个名称，服务器将响应任何提供的名称。响应中的模型名称将是列表中的第一个名称。如果未指定，模型名称将与--model参数相同。注意，如果提供多个名称，则指标的model_name标签内容也将使用第一个名称。 | |
| --qlora-adapter-name-or-path | QLoRA适配器的名称或路径。                                                                             |               |
| --engine-use-ray      | 是否使用Ray在单独的进程中启动LLM引擎作为服务器进程。                                                        | False         |
| --disable-log-requests | 禁用请求日志记录。                                                                                     | False         |
| --max-log-len         | 在日志中打印的最大提示字符数或提示ID数。                                                                      | Unlimited     |
