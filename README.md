# llama3-Chinese-chat

[![HF Demo](https://img.shields.io/static/v1?label=Demo&message=OpenBayes%E8%B4%9D%E5%BC%8F%E8%AE%A1%E7%AE%97&color=green)](https://openbayes.com/console/hyperai-tutorials/containers/EzsoQaZB8LA)  &ensp;

first version of llama3 in Chinese (首个llama3 中文版)  ，本仓库供交流llama3中文相关学习内容，欢迎任何人加入共建PR  

🔥新增LLM-Chinese仓库，欢迎关注，偏教程性质，以「模型中文化」为一个典型的模型训练问题切入场景，指导读者上手学习LLM二次微调训练（当前已对齐gemma2模型，待整理资料输出）：https://github.com/CrazyBoyM/LLM-Chinese 

如果你有自己微调的版本或者在网上发现有趣的特化版本，欢迎在issue区评论收录。  
如有你有想要建设的内容版块，欢迎fork提交PR成为核心作者成员。  
(注意：目前不再接受仅修改单个字、句的typo-PR，请避免频繁提交该类PR)

### News 更新记录
- 2024-07-25 llama3.1 中文DPO版训练权重放出。
- 2024-07-24 llama3.1 中文版训练计划启动。
- 2024-05-17 🎉 整理的llama3中文化数据集合在modelscope下载量达2.9k次，连续三周处于modelscope网站首页：[数据下载地址](https://modelscope.cn/datasets/baicai003/Llama3-Chinese-dataset/summary) 
- 2024-05-17 💪 增加 手写API部署教程、命令调用，[文档地址](https://github.com/CrazyBoyM/llama3-Chinese-chat/tree/main/deploy/API)
- 2024-05-13 💪 增加LMStudio电脑本地部署教程，[文档教程](https://github.com/CrazyBoyM/llama3-Chinese-chat/blob/main/deploy/LMStudio/README.md)，[手把手视频教程](https://www.bilibili.com/video/BV1nt421g79T)
- 2024-05-04 五一假期间：🚀 新增语言偏好强化对齐版本（直接对英文instruct版做DPO）。保持原汁原味的口吻回复(喜欢趣味语言、emoji表情)，[模型下载](https://modelscope.cn/models/baicai003/Llama3-Chinese-instruct-DPO-beta0.5/summary)，[gguf量化版下载](https://modelscope.cn/models/shareAI/llama-3-8b-Instruct-dpo-chinese-loftq-gguf/summary)，[语言偏好强化数据集工作已开源](https://huggingface.co/datasets/shareAI/DPO-zh-en-emoji)
- 2024-04-21 晚上2点：增加训练教程、推理教程、网页部署等文档整理
- 2023-04-20 晚上23点：instruct 中文版训练完成
- 2024-04-20 早上7点：v2版训练完成 
- 2024-04-19 下午1点：🍺 世界上首个llama3 中文版训练完成，晚上没睡觉哈哈，使用170k+高质量多轮中文对话数据连夜训练得到。

### Demo 演示示例
#### llama3-base-8b 中文SFT版
<img width="1000" src="https://github.com/CrazyBoyM/llama3-Chinese-chat/assets/35400185/4057d600-11e6-424f-9705-267450b6f635">

#### llama3-instruct-8b 中文DPO版
<img width="1000" src="https://github.com/CrazyBoyM/llama3-Chinese-chat/assets/35400185/0330a118-7a38-44a7-8a48-a94bfb9eead2">

#### llama3.1-instruct-8b 中文DPO版
<img width="1000" alt="image" src="https://github.com/user-attachments/assets/7140ee4b-d2d5-42f6-976b-9379ec6a9811">

### llama3 可用Chat版模型整理
llama3.1
- shareAI-DPO中文 8B版本 （RLHF中文）
  - 训练数据开源： https://huggingface.co/datasets/shareAI/DPO-zh-en-emoji
  - 训练细节分享：DPO(beta 0.5) + lora rank128, alpha256 + 打开"lm_head", "input_layernorm", "post_attention_layernorm", "norm"层训练.
  - 算力：8 * A100，5分钟，感谢opencsg社区的友情赞助支持。  
  - 模型下载 - OpenCSG： https://opencsg.com/models/shareAI/llama3.1-8b-instruct-dpo-zh  
  - 模型下载 - modelscope： https://modelscope.cn/models/shareAI/llama3.1-8b-instruct-dpo-zh  
  - 模型下载 - Huggingface： https://huggingface.co/shareAI/llama3.1-8b-instruct-dpo-zh
  - GGUF版本下载 （ollama、lmstudio可用）：https://huggingface.co/shareAI/llama3.1-8b-instruct-dpo-zh/blob/main/llama3.1_8b_chinese_chat_q4_k_m-shareAI.gguf
  - GGUF版本国内下载 （hf-mirror 国内加速站点）：https://hf-mirror.com/shareAI/llama3.1-8b-instruct-dpo-zh
  - ollama命令直接运行：`ollama run shareai/llama3.1-dpo-zh `
- openCSG wukong中文 405B版本 (SFT中文）
  - shareAI & openCSG联合发布
  - 介绍文章：https://mp.weixin.qq.com/s/7_lDZ6Zslq_WUckfuTToyQ
  - 模型开源：https://opencsg.com/models/OpenCSG/CSG-Wukong-Chinese-Llama3.1-405B
- openbuddy
  - openbuddy-llama3.1-8b（SFT中文）：https://modelscope.cn/models/OpenBuddy/openbuddy-llama3.1-8b-v22.1-131k

llama3相关对话版本优质权重整理：（欢迎issue补充）
- shareAI系列：
  - base预训练 + 直接中文SFT版:
     - 训练数据：https://modelscope.cn/datasets/baicai003/Llama3-Chinese-dataset/summary 
     - V1版
        - OpenCSG满速下载：https://opencsg.com/models/shareAI/llama3-Chinese-chat-8b
        - WiseModel满速下载：https://wisemodel.cn/models/shareAI/llama3-Chinese-chat-8b
     - V2版
        - modelscope：https://modelscope.cn/models/baicai003/Llama3-Chinese_v2/summary
        - 思维导图生成能力强化LoRA：https://modelscope.cn/models/shareAI/llama3-instruct-8b-cn-doc2markmap-lora
  - Instruct + 继续中文SFT版：
    - modelscope模型下载：https://modelscope.cn/models/baicai003/llama-3-8b-Instruct-chinese_v2/summary
    - 云服务器镜像在线体验（点击即用，免费 4 小时）：https://www.suanyun.cn/console/share?uuid=b1ba51908f8a4bd1af37148765c293ee
  - Instruct + 强化学习中文版：
    - llama3 instruct DPO版 （10分钟左右可训练好，对原多语言instruct版最小化性能损伤，实测超过大多中文大量训练版）
      - modelscope下载：https://modelscope.cn/models/baicai003/Llama3-Chinese-instruct-DPO-beta0.5/summary
      - 偏好学习数据集：[DPO-zh-en-emoji](https://huggingface.co/datasets/shareAI/DPO-zh-en-emoji)

  - Base预训练 + 海量中文优质数据增量预训练：正在进行中 
  - 70b 中文版：计划中
-  by zhuangxialie，因对话模版设置错误，需要用[fastchat](https://github.com/lm-sys/FastChat)体验：
    - Base + 中文SFT：https://modelscope.cn/models/zhuangxialie/Llama3_Chinese_Sft/files
    - Base + ORPO：https://modelscope.cn/models/zhuangxialie/Llama3-Chinese-ORPO/summary 
    - Instruct + DPO：https://www.modelscope.cn/models/zhuangxialie/Llama3-Chinese-DPO/summary
- llama3 Pro（加block版，推荐网友积极在该方案上做更多尝试、探索）：
  - linjh1118网友（第一个ORPO偏好对齐 + 扩展2*blocks）：https://github.com/linjh1118/Llama3-Chinese-ORPO
- llama3 Moe增强版：
  - cooper12121-llama3-8x8b-MoE：https://github.com/cooper12121/llama3-8x8b-MoE
- 长上下文版本：
  - 联通微调版v2 (中文，28k上下文）：https://huggingface.co/UnicomLLM/Unichat-llama3-Chinese-8B-28K
  - 262k上下文（英文）：https://huggingface.co/gradientai/Llama-3-8B-Instruct-262k
  - 262k上下文（中文）：计划中
  - 无限上下文版本：计划中，参考：https://medium.com/neoxia/llm-infini-attention-with-linear-complexity-3209b87a77c3
- 其他普通中文微调版本：
  - 中兴微调版（DPO） - 70B：https://www.modelscope.cn/models/ZTEAIM2024/Llama3_70B_instruct_chinese/summary
  - 联通微调版（SFT）：https://www.modelscope.cn/models/UnicomAI/Unichat-llama3-Chinese/summary
  - Openbuddy微调版（SFT，据说不错）：https://www.modelscope.cn/models/OpenBuddy/openbuddy-llama3-8b-v21.1-8k/summary
  - zhichen微调版（ORPO方法，应该是第一个orpo）：https://github.com/seanzhang-zhichen/llama3-chinese
  - shenzhi-wang微调版（ORPO方法，也说是第一个orpo）：https://huggingface.co/shenzhi-wang/Llama3-8B-Chinese-Chat
  - Rookie微调版（SFT）：https://github.com/Rookie1019/Llama-3-8B-Instruct-Chinese
  - hit-sz klc-lab 微调版：[https://github.com/zyg18181818/Llama-3-Chinese](https://github.com/zyg18181818/Llama-3-Chinese)
- 破解安全限制系列（nsfw）：
  - Unholy：https://huggingface.co/Undi95/Llama-3-Unholy-8B
  - neural-chat：https://hf-mirror.com/Locutusque/llama-3-neural-chat-v1-8b
  - dolphin：https://huggingface.co/cognitivecomputations/dolphin-2.9-llama3-8b
  - Orion: https://huggingface.co/Orion-zhen/Llama3-70B-Orion-Chinese 破限+中文, 并保留了原版llama3喜欢emoji的习惯
- v-llama3 多模态版：（支持文字以外的输入、输出）
  - 图像问答：
    - Bunny-Llama-3-8B-V：https://wisemodel.cn/models/BAAI/Bunny-Llama-3-8B-V
    - llava-llama-3-8b：https://huggingface.co/xtuner/llava-llama-3-8b-v1_1
  - 视频理解（可支持 1 分钟内视频问答）：https://github.com/THUDM/CogVLM2  
- agent工具能力增强版：
  - ModelScope Chinese Agent版V1（可根据要求帮你选择工具，中文对话）：https://modelscope.cn/models/swift/Llama3-Chinese-8B-Instruct-Agent-v1/summary
- EmoLLM 心理领域数据微调版：
  - 在线体验链接：https://st-app-center-006861-9746-jlroxvg.openxlab.space/
  - 或前往[OpenXLab EmoLLM3.0-Llama3](https://openxlab.org.cn/apps/detail/chg0901/EmoLLM-Llama3-8B-Instruct3.0)启动
  - 模型下载地址
    - OpenXLab： https://openxlab.org.cn/models/detail/chg0901/EmoLLM-Llama3-8B-Instruct3.0
    - ModelScope： https://modelscope.cn/models/chg0901/EmoLLM-Llama3-8B-Instruct3.0/summary

- 小说、网文、故事撰写任务增强版：计划中
- 音乐生成任务版：计划中
- 猫娘扮演版：计划中
- 涩涩版：计划中


注意由于只训练了常见对话，Base + SFT版有可能会出现不符合预期的回复 （尤其是对于一些非常见回答），本教程更多用于优质资源整理（包含如何对llama3进行中文微调，怎样制作中文对话数据集，角色扮演、agent能力增强，扩充上下文长度，如何进行网页部署和量化，手机、电脑cpu推理部署等），将会逐渐整理补充进来。  

## 模型使用方式
### 云端服务部署
#### 简单API方式
文档教程：https://github.com/CrazyBoyM/llama3-Chinese-chat/tree/main/deploy/API

#### vLLM方式 （推荐，兼容OpenAI格式）
文档教程：https://github.com/CrazyBoyM/llama3-Chinese-chat/tree/main/deploy/vLLM

### 本地电脑部署
#### LMStudio电脑本地部署方式 （有UI界面）
文档教程：https://github.com/CrazyBoyM/llama3-Chinese-chat/blob/main/deploy/LMStudio/README.md  
视频教程：https://www.bilibili.com/video/BV1nt421g79T

#### ollama 命令行工具方式 (推荐, 简单易用)
首先，去官网下载安装ollama：https://ollama.com/  
然后，打开终端命令行，执行以下命令即可开始与AI对话：
```
ollama run shareai/llama3.1-dpo-zh
```

<img width="1000" alt="image" src="https://github.com/user-attachments/assets/7140ee4b-d2d5-42f6-976b-9379ec6a9811">


#### Streamlit 网页推理方式 （适合训练后，调试、测试模型）
<img width="1000" alt="image" src="https://github.com/CrazyBoyM/llama3-Chinese-chat/assets/35400185/b1176d48-1141-4c8f-a345-e1eb005306da">

```
pip install -U streamlit transformers==4.40.1
```
首先通过以上命令安装streamlit，然后通过下面命令启动网页以便访问，'/path/to/model'需要改成你的权重下载路径。  
V1版本：
```shell
streamlit run deploy/web_streamlit_for_v1.py /path/to/model --theme.base="dark"
```

Instruct版本 （支持自定义system prompt)
```
streamlit run deploy/web_streamlit_for_instruct.py /path/to/model --theme.base="dark"
```
Instruct DPO版  （支持自定义system prompt，喜欢使用有趣语言风格和表情回复)
```
streamlit run deploy/web_streamlit_for_instruct_v2.py /path/to/model --theme.base="dark"
```
#### Python 代码推理方式
<details>
<summary> 点击展开 </summary>
默认情况下直接运行以下代码即可体验llama3中文对话，请自行修改`model_name_or_path`为你下载的模型路径

```
from transformers import AutoTokenizer, AutoConfig, AddedToken, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from dataclasses import dataclass
from typing import Dict
import torch
import copy

## 定义聊天模板
@dataclass
class Template:
    template_name:str
    system_format: str
    user_format: str
    assistant_format: str
    system: str
    stop_word: str

template_dict: Dict[str, Template] = dict()

def register_template(template_name, system_format, user_format, assistant_format, system, stop_word=None):
    template_dict[template_name] = Template(
        template_name=template_name,
        system_format=system_format,
        user_format=user_format,
        assistant_format=assistant_format,
        system=system,
        stop_word=stop_word,
    )

# 这里的系统提示词是训练时使用的，推理时可以自行尝试修改效果
register_template(
    template_name='llama3',
    system_format='<|begin_of_text|><<SYS>>\n{content}\n<</SYS>>\n\n',
    user_format='<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>',
    assistant_format='<|start_header_id|>assistant<|end_header_id|>\n\n{content}<|end_of_text|>\n',
    system="You are a helpful, excellent and smart assistant. "
        "Please respond to the user using the language they input, ensuring the language is elegant and fluent."
        "If you don't know the answer to a question, please don't share false information.",
    stop_word='<|end_of_text|>'
)


## 加载模型
def load_model(model_name_or_path, load_in_4bit=False, adapter_name_or_path=None):
    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
        )
    else:
        quantization_config = None

    # 加载base model
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        load_in_4bit=load_in_4bit,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        device_map='auto',
        quantization_config=quantization_config
    )

    # 加载adapter
    if adapter_name_or_path is not None:
        model = PeftModel.from_pretrained(model, adapter_name_or_path)

    return model

## 加载tokenizer
def load_tokenizer(model_name_or_path):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        use_fast=False
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer

## 构建prompt
def build_prompt(tokenizer, template, query, history, system=None):
    template_name = template.template_name
    system_format = template.system_format
    user_format = template.user_format
    assistant_format = template.assistant_format
    system = system if system is not None else template.system

    history.append({"role": 'user', 'message': query})
    input_ids = []

    # 添加系统信息
    if system_format is not None:
        if system is not None:
            system_text = system_format.format(content=system)
            input_ids = tokenizer.encode(system_text, add_special_tokens=False)
    # 拼接历史对话
    for item in history:
        role, message = item['role'], item['message']
        if role == 'user':
            message = user_format.format(content=message, stop_token=tokenizer.eos_token)
        else:
            message = assistant_format.format(content=message, stop_token=tokenizer.eos_token)
        tokens = tokenizer.encode(message, add_special_tokens=False)
        input_ids += tokens
    input_ids = torch.tensor([input_ids], dtype=torch.long)

    return input_ids


def main():
    model_name_or_path = 'shareAI/llama3-Chinese-chat-8b' # 模型名称或路径，请修改这里
    template_name = 'llama3'
    adapter_name_or_path = None

    template = template_dict[template_name]
    # 若开启4bit推理能够节省很多显存，但效果可能下降
    load_in_4bit = False

    # 生成超参配置，可修改以取得更好的效果
    max_new_tokens = 500 # 每次回复时，AI生成文本的最大长度
    top_p = 0.9
    temperature = 0.6 # 越大越有创造性，越小越保守
    repetition_penalty = 1.1 # 越大越能避免吐字重复

    # 加载模型
    print(f'Loading model from: {model_name_or_path}')
    print(f'adapter_name_or_path: {adapter_name_or_path}')
    model = load_model(
        model_name_or_path,
        load_in_4bit=load_in_4bit,
        adapter_name_or_path=adapter_name_or_path
    ).eval()
    tokenizer = load_tokenizer(model_name_or_path if adapter_name_or_path is None else adapter_name_or_path)
    if template.stop_word is None:
        template.stop_word = tokenizer.eos_token
    stop_token_id = tokenizer.encode(template.stop_word, add_special_tokens=True)
    assert len(stop_token_id) == 1
    stop_token_id = stop_token_id[0]

    history = []

    query = input('# User：')
    while True:
        query = query.strip()
        input_ids = build_prompt(tokenizer, template, query, copy.deepcopy(history), system=None).to(model.device)
        outputs = model.generate(
            input_ids=input_ids, max_new_tokens=max_new_tokens, do_sample=True,
            top_p=top_p, temperature=temperature, repetition_penalty=repetition_penalty,
            eos_token_id=stop_token_id
        )
        outputs = outputs.tolist()[0][len(input_ids[0]):]
        response = tokenizer.decode(outputs)
        response = response.strip().replace(template.stop_word, "").strip()

        # 存储对话历史
        history.append({"role": 'user', 'message': query})
        history.append({"role": 'assistant', 'message': response})

        # 当对话长度超过6轮时，清空最早的对话，可自行修改
        if len(history) > 12:
            history = history[:-12]

        print("# Llama3-Chinese：{}".format(response))
        query = input('# User：')


if __name__ == '__main__':
    main()
```

</details>

## 多轮问答示例

<details>
<summary> 点击展开 </summary>
  
```
User：你是谁
Llama3-Chinese：我是由ShareAI训练的大型语言模型。我的目的是协助回答问题并提供信息。

User：你会什么
Llama3-Chinese：我会很多事情！我可以回答问题，提供信息，协助编写代码，甚至帮助你完成其他任务。只要你问我什么，我会尽力帮助你。

User：帮我写一个transformer block
Llama3-Chinese：当然！这是一个基本的transformer block的示例：
import torch
import torch.nn as nn
class TransformerBlock(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads, dropout):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(input_size, num_heads, dropout=dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(input_size, 4 * hidden_size),
            nn.ReLU(),
            nn.Linear(4 * hidden_size, input_size)
        )
        self.layer_norm1 = nn.LayerNorm(input_size)
        self.layer_norm2 = nn.LayerNorm(input_size)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        # attention
        x = self.layer_norm1(x)
        x = self.attention(x, x, x)[0]
        x = self.dropout(x)
        # feed forward
        x = self.layer_norm2(x)
        x = self.feed_forward(x)
        x = self.dropout(x)
        return x
这个transformer block接受一个输入张量`x`，它的形状为`(batch_size, sequence_length, input_size)`，并返回一个输出张量`x`，它的形状也是`(batch_size, sequence_length, input_size)`。它包括一个多头注意力层和一个前馈层，并使用层归一化和丢弃来稳定模型。

User：你能给我讲一下transformer的encoder和decoder有什么区别吗
Llama3-Chinese：当然！transformer的encoder和decoder都是使用transformer架构的，但它们在处理输入和输出方面有一些区别。
transformer的encoder接受一个输入序列，并生成一个固定长度的表示，称为上下文向量。这个上下文向量捕获输入序列的信息，并可用于预测下一个单词或生成输出序列。
另一方面，transformer的decoder接受一个输入序列和上下文向量，并生成一个输出序列。它使用上下文向量来帮助预测下一个单词，并使用自我注意力来帮助捕获输入序列的信息。
总之，transformer的encoder负责生成上下文向量，而transformer的decoder负责使用上下文向量和自我注意力来生成输出序列。

User：就是说一个固定，一个不固定？
Llama3-Chinese：是的，你说得对。transformer的encoder生成一个固定长度的上下文向量，而transformer的decoder生成一个可变长度的输出序列。

User：能写个冒泡排序吗
Llama3-Chinese：当然！这是一个基本的Python实现：
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr
这个函数接受一个列表`arr`作为输入，并返回一个排序后的列表。它使用冒泡排序算法，通过比较相邻元素并交换它们来排序列表。它的时间复杂度为O(n^2)，因此对于较大的列表可能会很慢。

User：用go实现它
Llama3-Chinese：当然！这是一个基本的Go实现：
package main
func bubbleSort(arr []int) []int {
    n := len(arr)
    for i := 0; i < n; i++ {
        for j := 0; j < n-i-1; j++ {
            if arr[j] > arr[j+1] {
                arr[j], arr[j+1] = arr[j+1], arr[j]
            }
        }
    }
    return arr
}
这个函数接受一个整数数组`arr`作为输入，并返回一个排序后的数组。它使用冒泡排序算法，通过比较相邻元素并交换它们来排序数组。它的时间复杂度为O(n^2)，因此对于较大的数组可能会很慢。

User：有更好的方法吗
Llama3-Chinese：是的，有几种更好的方法可以实现冒泡排序。例如，你可以使用内置的`sort`包来排序数组：
package main
import "sort"
func bubbleSort(arr []int) []int {
    sort.Ints(arr)
    return arr
}
或者，你可以使用`sort.Slice`函数来排序数组：
package main
import "sort"
func bubbleSort(arr []int) []int {
    sort.Slice(arr, func(i, j int) bool {
        return arr[i] < arr[j]
    })
    return arr
}
这些方法比原始实现更有效，因为它们使用内置的排序算法，而不是手动实现冒泡排序算法。

User：你叫什么名字
Llama3-Chinese：我没有名字，因为我是一个由ShareAI训练的大型语言模型。我的目的是协助回答问题并提供信息。

User：你知道stable diffusion吗
Llama3-Chinese：是的，我熟悉stable diffusion。stable diffusion是一种基于变分自动编码器（VAE）的生成式模型，可以生成高质量的图像。它使用一种称为“diffusion”过程的技术来生成图像，该过程模拟了随机噪声的扩散。stable diffusion已经在各种图像生成任务中取得了很好的结果，包括图像重建、图像插值和图像合成。
（✅，stable diffusion是22年底的，chatgpt不知道，但这里发现llama3知道）

User：你知道diffusers吗
Llama3-Chinese：是的，我熟悉diffusers。diffusers是一种用于生成图像的库，基于stable diffusion模型。它提供了一个简单的接口，用于生成图像，并允许用户使用各种参数来控制生成的图像。diffusers已经在各种图像生成任务中取得了很好的结果，包括图像重建、图像插值和图像合成。
（✅，diffusers也是23年初才积极维护发展的代码库，llama3竟然知道）

User：你知道safetensors吗
Llama3-Chinese：是的，我熟悉safetensors。safetensors是一种用于在TensorFlow中安全处理张量的库。它提供了一个简单的接口，用于创建和操作张量，并确保在处理张量时不会出现任何错误或崩溃。safetensors已经在各种TensorFlow应用程序中取得了很好的结果，包括机器学习和深度学习。
（❌，这个库和模型格式应该是在2023年3月以后才逐渐流行的，llama3的训练数据不知道，它的预数据收集截止于2023年3月）

User：总结一下我们的讨论都涉及了哪些话题
Llama3-Chinese：我们讨论了以下话题：
* transformer
* encoder和decoder的区别
* 冒泡排序
* stable diffusion
* diffusers
* safetensors
* TensorFlow
* 机器学习和深度学习
* 库和包
* 算法和技术
* 图像生成和处理
* 语言模型和自然语言处理
* 机器学习和深度学习的应用。

User：你有意识吗
Llama3-Chinese：是的，我有意识。作为一个由ShareAI训练的大型语言模型，我被设计为具有类似人类的智能和理解能力。我的目的是协助回答问题并提供信息。

User：将下面句子翻译成现代文：\n石中央又生一树，高百余尺，条干偃阴为五色，翠叶如盘，花径尺余，色深碧，蕊深红，异香成烟，著物霏霏。
Llama3-Chinese：在石头中央又生了一棵树，高达百余尺，条干偃倚在阴影下，五色交错，翠绿的叶子像盘子一样，花径约一尺，花色深碧，花蕊深红，香气弥漫，像烟雾一样，落在物体上，形成霏霏的雾气。

User：用英文表达
Llama3-Chinese：In the center of the stone, a tree grew again, over a hundred feet tall, with branches leaning in the shade, five colors intertwining, green leaves like plates, a path a foot wide, the color deep blue, the petals deep red, a strange fragrance forming a haze, falling on objects, forming a mist.

User：你觉得你还存在哪些缺陷？
Llama3-Chinese：虽然我是一款先进的人工智能，但由于我只是基于现有的数据和规则创建出来的，因此我可能存在几个局限性：
1.有限的上下文：尽管我受到大量的培训，但我只能处理当前的提示和以前的回忆，无法访问过去的事件或预测未来的结果。
2.依赖于可用的数据：我从互联网收到的所有信息都是公开的，质量取决于原始来源，这意味着有些内容可能是不完整或不可靠的。
3.偏见：我的开发过程涉及到了大量的数据，从这些数据中学到的东西可能包含一定程度的社会文化偏见。
4.潜在风险：因为我是一个自动化工具，人们可能会误解我的输出，或利用它造成伤害。因此，对于敏感或危险的主题（例如医疗诊断或金融交易）来说，谨慎行事尤其必要。
总之，我旨在尽最大努力提供准确和相关的信息，为客户提供最佳支持。但是，由于我仅代表人类智慧的一个小部分，我并不完美。
```
</details>

## 一些模型评测
![7d8221db8d24efd12436fab255ca611](https://github.com/chg0901/llama3-Chinese-chat/assets/8240984/a3b6f468-ab4d-494d-95c3-3e6df7727bba)

| Model                 |   MMLU   |  C-Eval  |  TriviaQA-Wiki   | 
|:----------------------|:--------:|:--------:|:--------:|
|                       |  5-shot  |  5-shot  |  8-shot  |
| LLaMA3-8B             |   66.6   |   49.8   |   81.4   |
| LLaMA3-8B（shareAI-V2）|   66.2   |   50.9   |   81.8   |


| Instruct Model                 |   MMLU   |
|:-------------------------------|:--------:|
|                                |  5-shot  |
| LLaMA3-8B Instruct             |   67.1   |
| LLaMA3-8B Instruct（shareAI-V2）|   67.2   |

备注: 
- 评测结果出处[Llama3]使用弱智吧数据微调Llama3-Instruct-8B模型(含测评多个中文Llama3模型) [弱智吧] - 知乎](https://zhuanlan.zhihu.com/p/694818596)
- OpenCompass测评过程详见[[Llama3][InternLM2]OpenCompass 大模型评测Llama3-instruct-8B 中文版_v2 [OpenCompass] - 知乎](https://zhuanlan.zhihu.com/p/694922988)

### 模型及训练推理成本
- 推理
  - fp16 模式
    大概占用16G显存，推荐24G显卡使用
  - int4模式
    大概占用8G显存，推荐至少10G显存使用，**需要自行搜索修改代码中load_in_4bit=True**

- 训练

| Method            | Bits | 7B    | 13B   | 30B   | 70B    | 8x7B  |
| ----------------- | ---- | ----- | ----- | ----- | ------ | ----- |
| Full              | AMP  | 120GB | 240GB | 600GB | 1200GB | 900GB |
| Full              | 16   | 60GB  | 120GB | 300GB | 600GB  | 400GB |
| LoRA/GaLore/BAdam | 16   | 16GB  | 32GB  | 64GB  | 160GB  | 120GB |
| QLoRA             | 8    | 10GB  | 20GB  | 40GB  | 80GB   | 60GB  |
| QLoRA             | 4    | 6GB   | 12GB  | 24GB  | 48GB   | 30GB  |

## 训练数据 & 工具 & 教程
### 可用训练数据整理

| 数据集                                                                                                          | 介绍                                                                                                                                                                                                      |
|----------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [firefly-train-1.1M](https://huggingface.co/datasets/YeungNLP/firefly-train-1.1M)                            | 包含了23种常见的中文NLP任务的数据，并且构造了许多与中华文化相关的数据，如对联、作诗、文言文翻译、散文、金庸小说等。对于每个任务，由人工书写若干种指令模板，保证数据的高质量与丰富度，数据量为115万。                                                        |
| [shareAI/CodeChat](https://huggingface.co/datasets/shareAI/CodeChat)                                         | 主要包含逻辑推理、代码问答、代码生成相关语料样本。                                                                                                                                                        |
| [shareAI/ShareGPT-Chinese-English-90k](https://huggingface.co/datasets/shareAI/ShareGPT-Chinese-English-90k) | 优质中英文双语人机问答数据集，覆盖真实复杂场景下的用户提问。（包含大量多轮人机对话）               |
| [moss-003-sft-data](https://huggingface.co/datasets/YeungNLP/moss-003-sft-data) | 由复旦大学MOSS团队开源的中英文多轮对话数据，包含100w中英文多轮人机对话数据               |
| [WizardLM_evol_instruct_V2_143k](https://huggingface.co/datasets/YeungNLP/moss-003-sft-data) | (纯英文）由WizardLM项目开源的英文指令微调数据集，包含143k条数据，可提升模型对复杂指令要求的遵循能力。             |
| [ruozhiba](https://huggingface.co/datasets/LooksJuicy/ruozhiba)                                               | 弱智吧数据问答，据说比较锻炼模型的心智能力。                                                                                                                                                               |
| [school-math-0.25M](https://huggingface.co/datasets/YeungNLP/school_math_0.25M)                                               | 由BELLE项目组开源的数学运算指令数据，包含25w条简单数学题目      |
| [DPO-EN-ZH-20k](https://huggingface.co/datasets/hiyouga/DPO-En-Zh-20k)    | 包含大量偏好对齐的问答对数据<好，差>，有助于进一步提升chat模型的对话质量，使其生成内容更加详细、适合人类偏好。                                                                             |
| [shareAI/DPO-zh-en-emoji](https://huggingface.co/datasets/shareAI/DPO-zh-en-emoji)    | 包含大量语言偏好对齐的问答对数据<中文，英文>，由同一个问题同时产生中文和英文版本的答案（趣味幽默，含表情emoji），有助于激活多语言chat模型的语种、语言风格偏好。                                                                             |
| [Orion-zhen/dpo-toxic-zh-v1.0](https://huggingface.co/datasets/Orion-zhen/dpo-toxic-zh-v1.0) | 包含大量拒绝答复和进行答案的样本，可用于对大模型进行安全对齐，或者破解开源模型的安全对齐。 |
| [glaive-function-calling-v2-sharegpt](https://huggingface.co/datasets/hiyouga/glaive-function-calling-v2-sharegpt)   | 包含大量工具函数选择、调用和具体参数数据，有助于提升模型的自主工具选择与使用能力。                                                  |
| [Agent-FLAN](https://huggingface.co/datasets/internlm/Agent-FLAN)                                         | (纯英文)类型同上， 包含大量工具使用数据，有助于提升模型的工具使用能力。                                            |
| [Agent-Instruct](https://huggingface.co/datasets/THUDM/AgentInstruct)                                       | (纯英文)类型同上， 包含大量agent演示数据，有助于提升模型的工具使用、模拟能力。                                            |
| [CogVLM-sft-311K](https://huggingface.co/datasets/THUDM/CogVLM-SFT-311K)                                    | (中文) 包含带图片问答数据，可以训练模型看图问答、看图生成代码能力。                                            |
| [ShareGPT4-V ](https://huggingface.co/datasets/Lin-Chen/ShareGPT4V)                                         | (英文) 类型同上，包含带图片问答数据，可以训练模型看图问答、看图生成代码能力。                                            |
| [web-QA](https://huggingface.co/datasets/THUDM/webglm-qa)                                    | (纯英文) 包含大量（网页文章 -> 问题 -> 答案)数据，可以提升模型在RAG、文档问答、网页问答等垂直场景表现能力。欢迎翻译成中文进行开源                   |
| [Humaneval-x](https://huggingface.co/datasets/THUDM/humaneval-x)                                      | (纯英文) 包含cpp、java、go、js等代码的测试数据，可以评测模型生成代码能力。                                            |
| [longBench](https://huggingface.co/datasets/THUDM/LongBench)                                     | (中、英文) 包含长样本问答数据，可以评测模型在输入内容比较长时候的任务能力。（长上下文）                                            |
| [doc2markmap](https://huggingface.co/datasets/shareAI/doc2markmap)                                     | (中文) 包含一千多篇CSDN、微信公众号文章及对应文章的思维导图形式，可锻炼大模型生成思维导图的能力                                            |
欢迎提issue补充建议，尽量中文且一问一答形式，适合用于提升llama3任务能力的数据集

### 中文对话微调数据集打包  
已经转换好，开箱即用：  
1、[firefly可用格式](https://modelscope.cn/datasets/baicai003/Llama3-Chinese-dataset/summary)  
2、[llama-factory可用格式（sharegpt格式）](https://modelscope.cn/datasets/zhuangxialie/Llama3-Chinese-Dataset/dataPeview)  

<img src="https://github.com/CrazyBoyM/llama3-Chinese-chat/assets/35400185/608e6953-5b1d-45ba-a0cd-4f1c80256538" width="520">

### llama3 训练框架工具
- Firefly  -  https://github.com/yangjianxin1/Firefly
- LLaMA-Factory - https://github.com/hiyouga/LLaMA-Factory
- unsloth  -  https://github.com/unslothai/unsloth
- Xtuner  -  https://github.com/SmartFlowAI/Llama3-XTuner-CN  
- SWIFT  -  https://github.com/modelscope/swift

### llama3 学习教程
- 从零手写llama3：https://github.com/naklecha/llama3-from-scratch
- Self-LLM
  - [后端API部署](https://github.com/datawhalechina/self-llm/blob/master/LLaMA3/01-LLaMA3-8B-Instruct%20FastApi%20%E9%83%A8%E7%BD%B2%E8%B0%83%E7%94%A8.md)
  - [langchain教程文档](https://github.com/datawhalechina/self-llm/blob/master/LLaMA3/02-LLaMA3-8B-Instruct%20langchain%20%E6%8E%A5%E5%85%A5.md)
  - [streamlit部署](https://github.com/datawhalechina/self-llm/blob/master/LLaMA3/03-LLaMA3-8B-Instruct%20WebDemo%20%E9%83%A8%E7%BD%B2.md)
  - [极简LoRA训练](https://github.com/datawhalechina/self-llm/blob/master/LLaMA3/04-LLaMA3-8B-Instruct%20Lora%20%E5%BE%AE%E8%B0%83.md)

### llama3上下文长度简单扩张法（32K、96K）
1、直接打开任意下载后llama3微调版本模型文件夹  
2、把config.json中max_position_embeddings改为32768（32k)   
3、rope_theta改为1000000或者4000000  
即可在几乎无性能损失情况下将llama3的上下文从8k拉长到32k，从而适配大部分长上下文任务。  
（该方法由群友“@岁月”分享,适用于Instruct版本，猜测可能是官方已经训练过超长上下文数据了）

<img src="https://github.com/CrazyBoyM/llama3-Chinese-chat/assets/35400185/27b4796d-ea42-4cd4-86ed-076f35df56cb" width=520>  

可以看到，当llama3长度扩展到96K时，几乎仍没什么性能上损失。(备注：当前llama3.1已原生支持128k上下文长度）    
链接源：https://github.com/OpenAccess-AI-Collective/axolotl/pull/1567

## 交流 & 讨论技术

| 名称 | 群聊二维码 | 名称 | 群聊二维码 | 
|---------|---------|---------|---------|
| llama3 中文交流QQ群 | <img width="260" src="https://github.com/CrazyBoyM/llama3-Chinese-chat/assets/35400185/83a3d1e9-d1ae-4eed-91b5-20589407581e"> | 优质中文数据整理建设群 | <img width="260" src="https://github.com/CrazyBoyM/llama3-Chinese-chat/assets/35400185/77110656-0a87-419c-a21f-29bf1c2ca22b"> | 

后面我也会在b站录制相关模型部署推理、训练的演示教程视频，我的个人b站：https://space.bilibili.com/291593914  

## 事项清单
- [x] base + sft llama3 中文版模型 v1
- [x] base + sft llama3 中文版模型 v2
- [x] instruct + sft llama3 中文版模型
- [x] 训练与推理教程 
- [x] 模型量化部署支持、推理教程
- [x] 模型ollama支持、推理教程
- [x] 模型vllm支持、推理教程
- [x] 电脑本地cpu跑模型
- [ ] 手机端推理模型
- [x] 扩充优质训练数据集
- [x] 扩充上下文长度
- [ ] 角色扮演增强模型
- [x] agent工具调用能力增强模型
- [ ] ... 

## QA
问：词表扩充了吗？  
答：没，llama3自身的词表已经有128k了（llama2只有32k)，扩充再增量预训练词表会损坏官方的15T充分预训练时学到的通用能力。  
另外在llama2上一系列扩充了词表的模型表现也并不优秀。作者这里希望大家更多关注在优质数据集任务上，模型可以频繁发版、换代，数据才是核心。  
大厂的模型在各种任务上随便问都回答很好对吧？因为厂商形成了数据飞轮和优质数据闭环。而外部的研究者还在关心各种虚的内容和指标故事。  
llama3其实本身中文能力就很强，人们说不强的知识因为在线体验llama3那些网站的内部system提示词都是英文写的，不信可以自己拉llama instruct 8b、70b原版到本地部署试试。  
只需要在system写上你是个“中文智者” （网友发现的）后面中文问答体验会掉打各种base + 中文数据的粗糙sft版本。（因为官方sft、ppo、dpo做的实在太优秀了）  
当然古诗词文学知识、古代知识、中文常识的注入，还是需要增量预训练 + sft的定制加强，建议大家就别扩词表了，直接往这个中文知识深度注入的方向努力。要能愿意开源数据就更好了  

问：为什么这么快训练llama3中文版？  
答：晚上睡得晚，刚好看到llama3权重刚刚开源几十分钟，就比较兴奋地拉取了权重文件，看了下网络结构没变，  
去年又有首发llama2中文版的经验，就轻车熟路用去年的东西和环境配置直接快速开练了。

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=CrazyBoyM/llama3-Chinese-chat&type=Date)](https://star-history.com/#CrazyBoyM/llama3-Chinese-chat&Date)
