import json

def convert_entry(entry):
    conversation = {
        "human": entry["instruction"] + entry["input"],
        "assistant": entry["output"]
    }
    return conversation

def convert_jsonl(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f_input, open(output_file, 'w', encoding='utf-8') as f_output:
        for line in f_input:
            entry = json.loads(line)
            conversation = {"conversation": [convert_entry(entry)]}
            json.dump(conversation, f_output, ensure_ascii=False)
            f_output.write('\n')

if __name__ == "__main__":
    input_file = "./m-a-p/COIG-CQIA/zhihu/zhihu_expansion.jsonl"
    output_file = "zhihu_expansion.jsonl"

    convert_jsonl(input_file, output_file)
