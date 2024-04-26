def count_jsonl(input_file):
    max_len = 0
    min_len = 1e9
    total_len = 0
    count = 0
    with open(input_file, 'r', encoding='utf-8') as f_input:
        for line in f_input:
            if len(line) > max_len:
                max_len = len(line.strip().split())
            if len(line) < min_len:
                min_len = len(line.strip().split())
            total_len += len(line.strip().split())
            count += 1
        
        print("max_len: ", max_len)
        print("min_len: ", min_len)
        print("total_len: ", total_len)
        print(f"count: {count}")
        print(f"average_len: {total_len / count}")

if __name__ == "__main__":
    input_file = "./LongQLoRA-SFT_13k.jsonl"
    count_jsonl(input_file)
