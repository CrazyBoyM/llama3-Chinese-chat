import json
import random

def sample_jsonl(input_file, output_file):
    # 读取输入文件
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 随机采样1/3的数据
    sample_size = len(lines) // 3
    sampled_lines = random.sample(lines, sample_size)

    # 写入输出文件
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in sampled_lines:
            f.write(line)

if __name__ == "__main__":
    input_file = "input.jsonl"  # 输入文件路径
    output_file = "sampled_output.jsonl"  # 输出文件路径
    sample_jsonl(input_file, output_file)
