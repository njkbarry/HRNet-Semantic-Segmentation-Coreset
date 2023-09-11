#!/bin/bash

source venv/bin/activate

python -m torch.distributed.launch --nproc_per_node 1 tools/train.py --cfg experiments/pascal_ctx/gradient_approximation/HRNET/craig_ViT_05_a.yaml
python -m torch.distributed.launch --nproc_per_node 1 tools/train.py --cfg experiments/pascal_ctx/gradient_approximation/HRNET/craig_ViT_05_b.yaml
python -m torch.distributed.launch --nproc_per_node 1 tools/train.py --cfg experiments/pascal_ctx/gradient_approximation/HRNET/craig_ViT_05_c.yaml