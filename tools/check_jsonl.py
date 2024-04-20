# 用于检查数据集的每一条记录是否符合要求，不符合要求的将被删除

import json
import os

# 定义要检查的文件路径
file_path = "./dataset/v2/merged.jsonl"
# 定义用于存储有效行的列表
valid_lines = []

# 打开文件进行逐行检查
with open(file_path, "r") as file:
    for line_number, line in enumerate(file, start=1):
        line = line.strip()
        if line:
            try:
                data = json.loads(line)
                if "conversation" in data and data["conversation"]:
                    valid_lines.append(line)
                else:
                    print(f"删除第 {line_number} 行：无效的conversation")
            except json.JSONDecodeError:
                print(f"删除第 {line_number} 行：JSON解析错误")

# 删除原始文件
os.remove(file_path)

# 将有效行写入新文件
with open(file_path, "w") as file:
    for line in valid_lines:
        file.write(line + "\n")

print("检查完成并删除不符合要求的行。")
