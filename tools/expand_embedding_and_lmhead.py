# copied from https://github.com/HIT-SCIR/Chinese-Mixtral-8x7B/blob/main/models/init_embeddings.py
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import torch
from safetensors import safe_open
from transformers import AutoTokenizer


def init_embeddings_average(
        old_tokenizer,
        new_tokenizer,
        old_embeddings,
        old_lm_head,
        new_embeddings,
        new_lm_head,
):
    # set zh embeddings as average of old embeddings, but keep en embeddings unchanged

    old_vocab_size = old_tokenizer.vocab_size
    new_vocab_size = new_tokenizer.vocab_size

    for id in range(old_vocab_size, new_vocab_size):
        zh_token = new_tokenizer.decode([id])

        zh_token_old_ids = old_tokenizer(zh_token)["input_ids"]
        if len(zh_token_old_ids) == 0:
            print(f"WARNING: id = {id} zh_token = `{zh_token}`, cannot be tokenized by old tokenizer, using <unk> id")
            zh_token_old_ids = [0]  # unk
        zh_token_old_embeddings_avg = sum([old_embeddings[oid] for oid in zh_token_old_ids]) / len(zh_token_old_ids)
        zh_token_old_lm_head_avg = sum([old_lm_head[oid] for oid in zh_token_old_ids]) / len(zh_token_old_ids)
        new_embeddings[id] = zh_token_old_embeddings_avg
        new_lm_head[id] = zh_token_old_lm_head_avg


def draw(old_embeddings, new_embeddings, save):
    if not save:
        return

    plt.figure()
    plt.title(f"old embeddings[:, :128]")
    plt.xlabel("d_model[:128]")
    plt.ylabel("vocab_size")
    plt.imshow(old_embeddings[:, :128].to(torch.float16).numpy(), aspect="auto")
    plt.savefig(f"old-embeddings.png")

    plt.figure()
    plt.title(f"new embeddings[:, :128]")
    plt.xlabel("d_model[:128]")
    plt.ylabel("vocab_size")
    plt.imshow(new_embeddings[:, :128].to(torch.float16).numpy(), aspect="auto")
    plt.savefig(f"new-embeddings.png")


def main(
        old_tokenizer: str,
        new_tokenizer: str,
        num_shards: int,
        old_model: str,
        new_model: str,
        save_embedding_plots: bool = False,
):
    # load tokenizers
    old_tokenizer = AutoTokenizer.from_pretrained(old_tokenizer)
    new_tokenizer = AutoTokenizer.from_pretrained(new_tokenizer)
    new_vocab_size = len(new_tokenizer)  # __len__ = vocab_size + num_added_tokens

    # load embeddings and lm_head
    model_path_template = old_model + "/model-{index:05d}-of-{total:05d}.safetensors"
    model_dict = {}
    for i in range(1, num_shards + 1):
        shard_path = model_path_template.format(index=i, total=num_shards)
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for k in f.keys():
                model_dict[k] = f.get_tensor(k)

    # shape:
    #   old_embeddings: (vocab_size, d_model)
    #   old_lm_head:    (vocab_size, d_model)
    old_embeddings = model_dict["model.embed_tokens.weight"]
    old_lm_head = model_dict["lm_head.weight"]

    # create new embeddings and lm_head
    #   en: copy from old
    #   zh: init with zero
    new_embeddings = torch.zeros((new_vocab_size, old_embeddings.shape[1]), dtype=old_embeddings.dtype)
    new_lm_head = torch.zeros((new_vocab_size, old_lm_head.shape[1]), dtype=old_lm_head.dtype)
    new_embeddings[: old_embeddings.shape[0]] = old_embeddings.clone()
    new_lm_head[: old_lm_head.shape[0]] = old_lm_head.clone()

    init_embeddings_average(
        old_tokenizer,
        new_tokenizer,
        old_embeddings,
        old_lm_head,
        new_embeddings,
        new_lm_head,
    )

    draw(old_embeddings, new_embeddings, save_embedding_plots)

    model_dict["model.embed_tokens.weight"] = new_embeddings
    model_dict["lm_head.weight"] = new_lm_head

    torch.save(model_dict, Path(new_model) / "pytorch_model.bin")
    print(f"Done! `new_vocab_size` = {new_vocab_size}, please update `config.json` manually.")


if __name__ == "__main__":
    fire.Fire(main)
