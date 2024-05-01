def replace_keywords_and_remove_lines(input_file, output_file):
    # 定义关键词替换规则
    keywords = {
        "ChatGPT": "shareAI-GPT",
        "GPT3.5": "shareAI-GPT",
        "Gpt3.5": "shareAI-GPT",
        "gpt3.5": "shareAI-GPT",
        "GPT-3": "shareAI-GPT",
        "gpt-3": "shareAI-GPT",
        "OpenAI": "shareAI",
        "openAI": "shareAI",
        "openai": "shareAI",
    }
    # 定义要删除的关键词
    delete_keywords = ["无法", "不能", "can't", "can not"]

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
