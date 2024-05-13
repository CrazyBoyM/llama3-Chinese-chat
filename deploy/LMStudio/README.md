# LM Studio部署方式
<img width="1200" alt="image" src="https://github.com/CrazyBoyM/llama3-Chinese-chat/assets/35400185/7c692073-2103-41fa-b9aa-c4254a66ada0">

官方文档：https://lmstudio.ai/docs/welcome  
官网下载：https://lmstudio.ai/
### 设备支持列表
- Apple Silicon Mac (M1/M2/M3) with macOS 13.6 or newer
- Windows / Linux PC with a processor that supports AVX2 (typically newer PCs)
- 16GB+ of RAM is recommended. For PCs, 6GB+ of VRAM is recommended
- NVIDIA/AMD GPUs supported

### 部署&使用教程
1 、首先去[官网下载](https://lmstudio.ai/)，选择你的操作系统对应的安装版本，下载并安装。  
2 、去网上下载gguf格式的文件到本地，整理为三级文件夹结构 （如/models/shareAI/llama3-dpo-zh/xxx.gguf）。  
3 、进行模型导入、选择对话预设模板，进行加载使用。  
具体可参考视频演示： [b站视频教程](https://www.bilibili.com/video/BV1nt421g79T)

### API调用
首先，点击LM Studio的“Start Server”按钮打开api server，然后使用下面样例代码即可调用：
```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

completion = client.chat.completions.create(
  model="shareAI/llama3-dpo-zh", # 需修改为你的具体模型信息
  messages=[
    {"role": "system", "content": "you are a helpful bot."},
    {"role": "user", "content": "讲个笑话"}
  ],
  temperature=0.7,
  stop=["<|eot_id|>","<|eos_id|>"],
)

print(completion.choices[0].message)
```
