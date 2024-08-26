# This repo is used for finetuning mamba using deepspeed
> mamba original paper https://arxiv.org/ftp/arxiv/papers/2312/2312.00752.pdf
# Single GPU Use
```
python finetune_mamba.py --output_dir path/to/your/dir --model_name_or_path path/to/your/model
```
# Multiple GPU single node Use through deepspeed
```
deepspeed finetune_mamba.py --output_dir path/to/your/dir --model_name_or_path path/to/your/model --deepspeed path/to/your/deepspeed_config.json
```
# Multiple GPU Multiple node Use through deepspeed
```
torchrun --nproc_per_node=2 --nnode=4 finetune_mamba.py --output_dir path/to/your/dir --model_name_or_path path/to/your/model --deepspeed path/to/your/deepspeed_config.json
```
