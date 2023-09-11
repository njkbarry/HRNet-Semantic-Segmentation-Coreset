#!/bin/bash

source venv/bin/activate

# python -m torch.distributed.launch --nproc_per_node 1 tools/train.py --cfg experiments/pascal_ctx/stochastic_sampling_epsilon_experiment/HRNET/milo_ViT_05_0.1_a.yaml
# python -m torch.distributed.launch --nproc_per_node 1 tools/train.py --cfg experiments/pascal_ctx/stochastic_sampling_epsilon_experiment/HRNET/milo_ViT_05_0.1_b.yaml
# python -m torch.distributed.launch --nproc_per_node 1 tools/train.py --cfg experiments/pascal_ctx/stochastic_sampling_epsilon_experiment/HRNET/milo_ViT_05_0.1_c.yaml

# python -m torch.distributed.launch --nproc_per_node 1 tools/train.py --cfg experiments/pascal_ctx/stochastic_sampling_epsilon_experiment/HRNET/milo_ViT_05_0.01_a.yaml
# python -m torch.distributed.launch --nproc_per_node 1 tools/train.py --cfg experiments/pascal_ctx/stochastic_sampling_epsilon_experiment/HRNET/milo_ViT_05_0.01_b.yaml
# python -m torch.distributed.launch --nproc_per_node 1 tools/train.py --cfg experiments/pascal_ctx/stochastic_sampling_epsilon_experiment/HRNET/milo_ViT_05_0.01_c.yaml

# python -m torch.distributed.launch --nproc_per_node 1 tools/train.py --cfg experiments/pascal_ctx/stochastic_sampling_epsilon_experiment/HRNET/milo_ViT_05_0.00001_a.yaml
python -m torch.distributed.launch --nproc_per_node 1 tools/train.py --cfg experiments/pascal_ctx/stochastic_sampling_epsilon_experiment/HRNET/milo_ViT_05_0.00001_b.yaml
python -m torch.distributed.launch --nproc_per_node 1 tools/train.py --cfg experiments/pascal_ctx/stochastic_sampling_epsilon_experiment/HRNET/milo_ViT_05_0.00001_c.yaml

