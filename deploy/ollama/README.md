# ollama 部署语言模型
## 简单介绍 ollama
ollama是一个完全借鉴docker思想开发的软件，它底层封装了llama.cpp的代码，上层提供了命令行工具，能够便捷地提供终端对话推理、本地api部署等能力，用户无需关心复杂的对话模型参数设置与权重下载管理等问题。

## 上手使用 ollama
首先，去官网下载安装ollama（非常简单 容易安装）：https://ollama.com/  
然后，打开命令行终端，执行以下命令即可开始与AI对话：
```
ollama run shareai/llama3.1-dpo-zh
```

<img width="1000" alt="image" src="https://github.com/user-attachments/assets/7140ee4b-d2d5-42f6-976b-9379ec6a9811">

## 通过API连接ollama (兼容openai格式)
运行上面命令后，ollama会默认启动一个api服务，可以通过以下命令进行调用测试：  
```shell
curl http://localhost:11434/v1/chat/completions -d '{
  "model": "shareai/llama3.1-dpo-zh",
  "messages": [
    {
      "role": "user",
      "content": "讲个笑话?"
    }
  ],
  "stream": false
}'
```

## 更多
可以参考ollama中文文档：https://ollama.fan/reference/api/
