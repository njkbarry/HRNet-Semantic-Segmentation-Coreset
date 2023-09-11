#!/bin/bash

source venv/bin/activate

# python -m torch.distributed.launch --nproc_per_node 1 tools/train.py --cfg experiments/pascal_ctx/static_random_profiling/HRNET/static_random_03_a.yaml
# python -m torch.distributed.launch --nproc_per_node 1 tools/train.py --cfg experiments/pascal_ctx/static_random_profiling/HRNET/static_random_03_b.yaml
# python -m torch.distributed.launch --nproc_per_node 1 tools/train.py --cfg experiments/pascal_ctx/static_random_profiling/HRNET/static_random_03_c.yaml

# python -m torch.distributed.launch --nproc_per_node 1 tools/train.py --cfg experiments/pascal_ctx/static_random_profiling/HRNET/static_random_05_a.yaml
python -m torch.distributed.launch --nproc_per_node 1 tools/train.py --cfg experiments/pascal_ctx/static_random_profiling/HRNET/static_random_05_b.yaml
python -m torch.distributed.launch --nproc_per_node 1 tools/train.py --cfg experiments/pascal_ctx/static_random_profiling/HRNET/static_random_05_c.yaml

python -m torch.distributed.launch --nproc_per_node 1 tools/train.py --cfg experiments/pascal_ctx/static_random_profiling/HRNET/static_random_08_a.yaml
python -m torch.distributed.launch --nproc_per_node 1 tools/train.py --cfg experiments/pascal_ctx/static_random_profiling/HRNET/static_random_08_b.yaml
python -m torch.distributed.launch --nproc_per_node 1 tools/train.py --cfg experiments/pascal_ctx/static_random_profiling/HRNET/static_random_08_c.yaml