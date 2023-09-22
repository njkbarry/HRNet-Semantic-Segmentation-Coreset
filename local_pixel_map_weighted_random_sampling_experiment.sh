#!/bin/bash

source venv/bin/activate

python -m torch.distributed.launch --nproc_per_node 1 tools/train.py --cfg experiments/pascal_ctx/pixel_map_weighted_random_sampling_experiment/HRNET/baseset_00113_08_a.yaml
python -m torch.distributed.launch --nproc_per_node 1 tools/train.py --cfg experiments/pascal_ctx/pixel_map_weighted_random_sampling_experiment/HRNET/baseset_00113_08_b.yaml
python -m torch.distributed.launch --nproc_per_node 1 tools/train.py --cfg experiments/pascal_ctx/pixel_map_weighted_random_sampling_experiment/HRNET/baseset_00113_08_c.yaml