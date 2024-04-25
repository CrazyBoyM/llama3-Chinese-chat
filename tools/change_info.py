def replace_keywords_and_remove_lines(input_file, output_file):
    # 定义关键词替换规则
    keywords = {
        "ChatGPT": "LLama3-Chinese",
        "GPT3.5": "LLama3-Chinese",
        "GPT-3": "LLama3-Chinese",
    }
    # 定义要删除的关键词
    delete_keywords = ["无法", "OpenAI", "抱歉", "Sorry", "对不起", "道德"]

    with open(input_file, 'r') as input_f, open(output_file, 'w') as output_f:
        for line in input_f:
            if any(keyword in line for keyword in delete_keywords):
                continue  # 如果包含要删除的关键词，则跳过该行

            # 逐个检查关键词并替换
            for keyword, replacement in keywords.items():
                line = line.replace(keyword, replacement)
            
            # 将替换后的行写入输出文件
            output_f.write(line)

    print("关键词替换并删除行完成！")


# 指定输入文件和输出文件的路径
input_file_path = "./dataset/v2/merged.jsonl"
output_file_path = "./train_v2.jsonl"

# 调用函数进行关键词替换和删除行
replace_keywords_and_remove_lines(input_file_path, output_file_path)
