# 可用于提升模型效果的一种技巧，将最近检查点权重融合, 与RWKV Merge_Lora原理类似，都是对权重的提取合并
import torch

w_a = torch.load('./checkpoint-700/adapter_model.bin', map_location=torch.device('cpu'))
w_b = torch.load('./checkpoint-800/adapter_model.bin', map_location=torch.device('cpu'))
w_c = {}

for k in w_a.keys():
    try:
        w_c[k] = w_a[k] * 0.7 + w_b[k] * 0.3
    except:
        print(k)
    
for k in w_a.keys():
    if k not in w_c.keys():
        w_c[k] = w_a[k]

torch.save(w_c, 'adapter_model_merged.bin')
