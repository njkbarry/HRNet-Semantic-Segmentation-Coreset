#!/bin/bash

source venv/bin/activate

# python -m torch.distributed.launch --nproc_per_node 1 tools/train.py --cfg experiments/pascal_ctx/adaptive_random_profiling/HRNET/ar_03_a.yaml
# python -m torch.distributed.launch --nproc_per_node 1 tools/train.py --cfg experiments/pascal_ctx/adaptive_random_profiling/HRNET/ar_03_b.yaml
# python -m torch.distributed.launch --nproc_per_node 1 tools/train.py --cfg experiments/pascal_ctx/adaptive_random_profiling/HRNET/ar_03_c.yaml


# python -m torch.distributed.launch --nproc_per_node 1 tools/train.py --cfg experiments/pascal_ctx/adaptive_random_profiling/HRNET/ar_05_a.yaml
# python -m torch.distributed.launch --nproc_per_node 1 tools/train.py --cfg experiments/pascal_ctx/adaptive_random_profiling/HRNET/ar_05_b.yaml
# python -m torch.distributed.launch --nproc_per_node 1 tools/train.py --cfg experiments/pascal_ctx/adaptive_random_profiling/HRNET/ar_05_c.yaml


# python -m torch.distributed.launch --nproc_per_node 1 tools/train.py --cfg experiments/pascal_ctx/adaptive_random_profiling/HRNET/ar_08_a.yaml
# python -m torch.distributed.launch --nproc_per_node 1 tools/train.py --cfg experiments/pascal_ctx/adaptive_random_profiling/HRNET/ar_08_b.yaml


python -m torch.distributed.launch --nproc_per_node 1 tools/train.py --cfg experiments/pascal_ctx/adaptive_random_profiling/HRNET/ar_05_d.yaml
python -m torch.distributed.launch --nproc_per_node 1 tools/train.py --cfg experiments/pascal_ctx/adaptive_random_profiling/HRNET/ar_05_e.yaml
python -m torch.distributed.launch --nproc_per_node 1 tools/train.py --cfg experiments/pascal_ctx/adaptive_random_profiling/HRNET/ar_05_f.yaml

python -m torch.distributed.launch --nproc_per_node 1 tools/train.py --cfg experiments/pascal_ctx/adaptive_random_profiling/HRNET/ar_08_c.yaml