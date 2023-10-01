#!/bin/bash

source venv/bin/activate

python -m torch.distributed.launch --nproc_per_node 1 tools/train.py --cfg experiments/pascal_ctx/uncertainty_experiment/lcu_entr_05_a.yaml
python -m torch.distributed.launch --nproc_per_node 1 tools/train.py --cfg experiments/pascal_ctx/uncertainty_experiment/lcu_entr_05_b.yaml
python -m torch.distributed.launch --nproc_per_node 1 tools/train.py --cfg experiments/pascal_ctx/uncertainty_experiment/lcu_entr_05_c.yaml

python -m torch.distributed.launch --nproc_per_node 1 tools/train.py --cfg experiments/pascal_ctx/uncertainty_experiment/lcu_entr_rev_05_a.yaml
python -m torch.distributed.launch --nproc_per_node 1 tools/train.py --cfg experiments/pascal_ctx/uncertainty_experiment/lcu_entr_rev_05_b.yaml
python -m torch.distributed.launch --nproc_per_node 1 tools/train.py --cfg experiments/pascal_ctx/uncertainty_experiment/lcu_entr_rev_05_c.yaml

python -m torch.distributed.launch --nproc_per_node 1 tools/train.py --cfg experiments/pascal_ctx/uncertainty_experiment/lcu_prob_05_a.yaml
python -m torch.distributed.launch --nproc_per_node 1 tools/train.py --cfg experiments/pascal_ctx/uncertainty_experiment/lcu_prob_05_b.yaml
python -m torch.distributed.launch --nproc_per_node 1 tools/train.py --cfg experiments/pascal_ctx/uncertainty_experiment/lcu_prob_05_c.yaml

python -m torch.distributed.launch --nproc_per_node 1 tools/train.py --cfg experiments/pascal_ctx/uncertainty_experiment/lcu_prob_rev_05_a.yaml
python -m torch.distributed.launch --nproc_per_node 1 tools/train.py --cfg experiments/pascal_ctx/uncertainty_experiment/lcu_prob_rev_05_b.yaml
python -m torch.distributed.launch --nproc_per_node 1 tools/train.py --cfg experiments/pascal_ctx/uncertainty_experiment/lcu_prob_rev_05_c.yaml

