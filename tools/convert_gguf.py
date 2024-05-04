from unsloth import FastLanguageModel
import torch
max_seq_length = 8096 
dtype = None 
load_in_4bit = False # Use 4bit quantization to reduce memory usage. Can be False.

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "./Llama3-Chinese-instruct-DPO-beta0.5-loftq", # 改成模型路径
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

# model.save_pretrained_gguf("llama3-Chinese-chat-8b_gguf_Q8_0", tokenizer,)
model.save_pretrained_gguf("model", tokenizer, quantization_method = "q4_k_m") # q5、q8会更好
