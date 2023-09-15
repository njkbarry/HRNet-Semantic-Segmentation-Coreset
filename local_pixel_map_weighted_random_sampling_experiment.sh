#!/bin/bash

source venv/bin/activate

python -m torch.distributed.launch --nproc_per_node 1 tools/train.py --cfg experiments/pascal_ctx/pixel_map_weighted_random_sampling_experiment/HRNET/baseset_003_08_a.yaml
python -m torch.distributed.launch --nproc_per_node 1 tools/train.py --cfg experiments/pascal_ctx/pixel_map_weighted_random_sampling_experiment/HRNET/baseset_003_08_b.yaml
python -m torch.distributed.launch --nproc_per_node 1 tools/train.py --cfg experiments/pascal_ctx/pixel_map_weighted_random_sampling_experiment/HRNET/baseset_003_08_c.yaml

# python -m torch.distributed.launch --nproc_per_node 1 tools/train.py --cfg experiments/pascal_ctx/pixel_map_weighted_random_sampling_experiment/HRNET/invtrainprop_05_a.yaml
# python -m torch.distributed.launch --nproc_per_node 1 tools/train.py --cfg experiments/pascal_ctx/pixel_map_weighted_random_sampling_experiment/HRNET/invtrainprop_05_b.yaml
python -m torch.distributed.launch --nproc_per_node 1 tools/train.py --cfg experiments/pascal_ctx/pixel_map_weighted_random_sampling_experiment/HRNET/invtrainprop_05_c.yaml