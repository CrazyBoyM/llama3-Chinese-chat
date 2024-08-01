# API部署
## 一、简单手写版本
### 代码准备
首先安装包依赖：
```shell
pip install -U transformers fastapi accelerate
```
然后运行easy_server_demo.py，以下为代码示例：
```
import uvicorn
import torch
from transformers import pipeline, AutoTokenizer
from fastapi import FastAPI, Request

app = FastAPI()

@app.post("/")
async def create_item(request: Request):
    global pipe
    data = await request.json()
    prompt = data.get('prompt')
    print(prompt)

    messages = [
            {
                "role": "system",
                "content": "你是一个超级智者，名字叫shareAI-llama3，拥有优秀的问题解答能力。",
            },
            {"role": "user", "content": prompt}
    ]
    
    response = pipe(messages)
    # breakpoint()
    print(response)
    answer = {
        "response": response[-1]["content"],
        "status": 200,
    }
    return answer


if __name__ == '__main__':
    model_name_or_path = '/openbayes/home/baicai003/Llama3-Chinese-instruct-DPO-beta0___5'
    # 这里的模型路径替换为你本地的完整模型存储路径 （一般从huggingface或者modelscope上下载到）
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)
    pipe = pipeline(
        "conversational", 
        model_name_or_path, 
        torch_dtype=torch.float16, 
        device_map="auto", 
        max_new_tokens=512, 
        do_sample=True,
        top_p=0.9, 
        temperature=0.6, 
        repetition_penalty=1.1,
        eos_token_id=tokenizer.encode('<|eot_id|>')[0]
    )
    # 如果是base+sft模型需要替换<|eot_id|>为<|end_of_text|>，因为llama3 base模型里没有训练<|eot_id|>这个token

    uvicorn.run(app, host='0.0.0.0', port=9009) # 这里的端口替换为你实际想要监听的端口
```

上面代码中使用了transformers的[pipeline](https://github.com/huggingface/transformers/blob/main/docs/source/en/conversations.md)进行实现，具体来说，它相当于以下操作：
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 输入内容
chat = [
    {"role": "system", "content": "You are a sassy, wise-cracking robot as imagined by Hollywood circa 1986."},
    {"role": "user", "content": "Hey, can you tell me any fun things to do in New York?"}
]

# 1: 加载模型、分词器
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", device_map="auto", torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

# 2: 使用对话模板
formatted_chat = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
print("Formatted chat:\n", formatted_chat)

# 3: 将对话内容转为token (也可以在上一步直接开启tokenize=True)
inputs = tokenizer(formatted_chat, return_tensors="pt", add_special_tokens=False)

# 把tokens转移到GPU或者CPU上
inputs = {key: tensor.to(model.device) for key, tensor in inputs.items()}
print("Tokenized inputs:\n", inputs)

# 4: 使用模型生成一段文本
outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.)
print("Generated tokens:\n", outputs)

# 5: 把生成结果从离散token变为文本
decoded_output = tokenizer.decode(outputs[0][inputs['input_ids'].size(1):], skip_special_tokens=True)
print("Decoded output:\n", decoded_output)
```
### 调用测试
命令：
```shell
curl -X POST "http://127.0.0.1:9009"  -H 'Content-Type: application/json'  -d '{"prompt": "先有鸡 还是先有蛋"}'
```
通过在终端执行以上命令即可调用，返回：
```json
{
  "response":"😂哈哈，老问题！🤯\n\n这个问题被称为“鸡和蛋的循环论证”，是指两个概念相互依赖、无法确定优先顺序的逻辑悖论。 🐓🥚\n\n从生物学角度来看，鸡蛋是鸟类的一种生殖方式，鸡的雏化过程中需要蛋孵化，而鸡又是蛋的产物。 👀\n\n那么，问题来了：如果说先有蛋，那么鸡就不存在了，因为鸡是蛋孵化出来的；如果说先有鸡，那么蛋就不存在了，因为鸡没有蛋来孵化。 🤔\n\n这个问题可以从多个方面去理解：\n\n1️⃣从演化角度来说，生物进化是一个漫长的过程，鸡和蛋都是自然选择和适应环境的结果。 🌳\n\n2️⃣从定义角度来说，鸡和蛋都是相互依赖的概念，鸡就是蛋孵化出来的，蛋就是鸡产出的。 🤝\n\n3️⃣从哲学角度来说，这个问题涉及到时间概念和空间概念的关系，时间和空间都不是线性的，存在某种程度的相对性。 🕰️\n\n总之，鸡和蛋的先后关系只是一个逻辑上的循环论证，实际上我们不需要担心这个问题，因为它们都是生物界中的常态现象！ 😊",
  "status":200
}
```
如果你需要在其他开发语言中使用，可以用gpt将调用命令转换为其他语言版本(如python、java、php）

## 二、 OpenAI格式版本
请参考vLLM部署教程：https://github.com/CrazyBoyM/llama3-Chinese-chat/tree/main/deploy/vLLM
