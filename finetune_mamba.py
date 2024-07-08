import json

from transformers import AutoModel, AutoTokenizer, HfArgumentParser

from transformers import AutoTokenizer, AutoModelForCausalLM


from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from dataclasses import dataclass, field
import pathlib
from typing import Dict, Optional, Sequence
import transformers
from transformers import Trainer
from transformers.trainer_pt_utils import LabelSmoother
import torch
from torch.utils.data import Dataset




local_rank = None
from collections import namedtuple

def rank0_print(*args):
    if local_rank == 0:
        print(*args)
@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="../mamba-700m")
@dataclass
class DataArguments:
    data_path: str = field(
        default='/root/autodl-tmp/LLM_EXPERIMENT/mamba-main/data/train_8k.json', metadata={"help": "Path to the training data."}
    )
    eval_data_path: str = field(
        default='/root/autodl-tmp/LLM_EXPERIMENT/mamba-main/data/eva_2k.json', metadata={"help": "Path to the evaluation data."}
    )
    lazy_preprocess: bool = False
@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=1024,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    per_device_train_batch_size: int = field(default=1)
    per_device_eval_batch_size: int = field(default=1)
    evaluation_strategy: str = field(default="steps")
    logging_steps: int = field(default=1)
    num_train_epochs: int = field(default=3)
    lr_scheduler_type: str = field(default="cosine")
    learning_rate: float = field(default=2e-5)
    group_by_length: bool = field(default=True)
    bf16: bool = field(default=True)
    gradient_accumulation_steps: int = field(default=1)
    save_strategy: str = field(default="steps")
    save_steps: int = field(default=1200)
    eval_steps: int = field(default=1200)
    # tf32: bool = field(default=True)
    deepspeed: str = field(default="playground/deepspeed_config_s3.json")
    output_dir: Optional[str] = field(default="./output")


def resize_token_embeddings(model, new_num_tokens):
    import torch.nn as nn

    old_embeddings = model.backbone.embedding
    old_num_tokens, old_embedding_dim = old_embeddings.weight.size()
    new_embeddings = nn.Embedding(
        new_num_tokens,
        old_embedding_dim,
        device=old_embeddings.weight.device,
        dtype=old_embeddings.weight.dtype,
    )
    nn.init.normal_(new_embeddings.weight, std=0.02)
    n = min(old_num_tokens, new_num_tokens)
    new_embeddings.weight.data[:n, :] = old_embeddings.weight.data[:n, :]
    model.backbone.embedding = new_embeddings

    model.tie_weights()
def forward_with_loss(self, input_ids,attention_mask=None, inference_params=None, num_last_tokens=0, labels=None):
    """
    "position_ids" is just to be compatible with Transformer generation. We don't use it.
    num_last_tokens: if > 0, only return the logits for the last n tokens
    """
    hidden_states = self.backbone(input_ids, inference_params=inference_params)
    if num_last_tokens > 0:
        hidden_states = hidden_states[:, -num_last_tokens:]
    lm_logits = self.lm_head(hidden_states)

    # Source: https://github.com/huggingface/transformers/blob/80377eb018c077dba434bc8e7912bcaed3a64d09/src/transformers/models/llama/modeling_llama.py#L1196
    from torch.nn import CrossEntropyLoss
    if labels is not None:
        logits = lm_logits
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        # shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_logits = shift_logits.view(-1, self.backbone.embedding.weight.size()[0])
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)
        return (loss,)
    else:
        CausalLMOutput = namedtuple("CausalLMOutput", ["logits"])
        return CausalLMOutput(logits=lm_logits)
MambaLMHeadModel.forward = forward_with_loss
# def collate(elements):
#     tokenlist=[e["input_ids"] for e in elements]
#     tokens_maxlen=max([len(t) for t in tokenlist])
#
#     input_ids,labels = [],[]
#     for tokens in tokenlist:
#         pad_len=tokens_maxlen-len(tokens)
#
#         # pad input_ids with pad_token, labels with ignore_index (-100)
#         input_ids.append( tokens + [tokenizer.pad_token_id]*pad_len )
#         labels.append( tokens + [-100]*pad_len )
#     batch={
#         "input_ids": torch.tensor(input_ids),
#         "labels": torch.tensor(labels),
#     }
#     return batch
import copy
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}
def InstructionDataset(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    IGNORE_TOKEN_ID = -100  # The default setting in CrossEntropyLoss
    examples = []
    prompts = []
    for ann in sources:
        if ann.get("input", "") == "":
            prompt = PROMPT_DICT["prompt_no_input"].format_map(ann)
        else:
            prompt = PROMPT_DICT["prompt_input"].format_map(ann)
        example = prompt + ann["output"]
        examples.append(example)
        prompts.append(prompt)
    input_ids = tokenizer(
        examples,
        padding="max_length",
        return_tensors="pt",
        max_length=tokenizer.model_max_length,
        truncation=True,
        add_special_tokens=False
    ).input_ids
    labels = copy.deepcopy(input_ids)
    for prompt,example, target in zip(prompts,input_ids, labels):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())
        cur_len = len(tokenizer(prompt).input_ids) - 1
        target[:cur_len] = IGNORE_TOKEN_ID
        cur_len = int(example.ne(tokenizer.pad_token_id).sum())
        target[cur_len:] = IGNORE_TOKEN_ID
        if False:  # Inspect and check the correctness of masking
            z = target.clone()
            z = torch.where(z == IGNORE_TOKEN_ID, tokenizer.unk_token_id, z)
            rank0_print(tokenizer.decode(z))

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_TOKEN_ID
                rank0_print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )
    return {
        "input_ids": input_ids,
        "labels": labels,
        # "attention_mask":input_ids.ne(tokenizer.pad_token_id),
    }
class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()

        rank0_print("Formatting inputs...")
        # sources = [example["conversations"] for example in raw_data]
        sources = [example for example in raw_data]
        # data_dict = preprocess(sources, tokenizer)
        data_dict = InstructionDataset(sources,tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        # self.attention_mask = data_dict["attention_mask"]


    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i],
            # attention_mask=self.attention_mask[i],
        )
def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset_cls = SupervisedDataset
    rank0_print("Loading data...")

    train_json = json.load(open(data_args.data_path, "r"))
    train_dataset = dataset_cls(train_json, tokenizer=tokenizer)

    if data_args.eval_data_path:
        eval_json = json.load(open(data_args.eval_data_path, "r"))
        eval_dataset = dataset_cls(eval_json, tokenizer=tokenizer)
    else:
        eval_dataset = None

    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)
device = "cuda"
dtype = torch.bfloat16
def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model = MambaLMHeadModel.from_pretrained(
        "/root/autodl-tmp/LLM_EXPERIMENT/mamba-130m",
        dtype=torch.bfloat16,
        # device="cuda",
    )
    tokenizer = AutoTokenizer.from_pretrained("/root/autodl-tmp/LLM_EXPERIMENT/gpt-neox-20b",
        model_max_length = training_args.model_max_length,
        padding_side = "right",
    )
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.add_tokens(["<PAD>"])
    tokenizer.add_tokens(["<|im_start|>"])
    tokenizer.add_special_tokens(dict(eos_token="<|im_end|>"))
    tokenizer.pad_token = "<PAD>"
    tokenizer.eos_token = "<|im_end|>"
    resize_token_embeddings(model, len(tokenizer))
    rank0_print("Loading data...")
    data_module = make_supervised_data_module(tokenizer,data_args)

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **data_module
    )
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        print("start training------")
        trainer.train()
    model.config.use_cache = True
    trainer.save_model()
    trainer.save_state()

if __name__ == "__main__":
    main()