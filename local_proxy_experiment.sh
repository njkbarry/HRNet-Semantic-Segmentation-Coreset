#!/bin/bash

source venv/bin/activate

python -m torch.distributed.launch --nproc_per_node 1 tools/train.py --cfg /home/nickbarry/Documents/MsC-DS/Data_Science_Research_Project/Coresets/Repositories/HRNet-Semantic-Segmentation-Coreset/experiments/pascal_ctx/proxy_experiment/HRNET/0.8/ar_08_a.yaml
python -m torch.distributed.launch --nproc_per_node 1 tools/train.py --cfg /home/nickbarry/Documents/MsC-DS/Data_Science_Research_Project/Coresets/Repositories/HRNet-Semantic-Segmentation-Coreset/experiments/pascal_ctx/proxy_experiment/HRNET/0.8/ar_08_b.yaml
python -m torch.distributed.launch --nproc_per_node 1 tools/train.py --cfg /home/nickbarry/Documents/MsC-DS/Data_Science_Research_Project/Coresets/Repositories/HRNet-Semantic-Segmentation-Coreset/experiments/pascal_ctx/proxy_experiment/HRNET/0.8/ar_08_c.yaml

python -m torch.distributed.launch --nproc_per_node 1 tools/train.py --cfg /home/nickbarry/Documents/MsC-DS/Data_Science_Research_Project/Coresets/Repositories/HRNet-Semantic-Segmentation-Coreset/experiments/pascal_ctx/proxy_experiment/HRNET/0.8/milo_oracle_spat_fronerbius_08_a.yaml
python -m torch.distributed.launch --nproc_per_node 1 tools/train.py --cfg /home/nickbarry/Documents/MsC-DS/Data_Science_Research_Project/Coresets/Repositories/HRNet-Semantic-Segmentation-Coreset/experiments/pascal_ctx/proxy_experiment/HRNET/0.8/milo_oracle_spat_fronerbius_08_b.yaml
python -m torch.distributed.launch --nproc_per_node 1 tools/train.py --cfg /home/nickbarry/Documents/MsC-DS/Data_Science_Research_Project/Coresets/Repositories/HRNet-Semantic-Segmentation-Coreset/experiments/pascal_ctx/proxy_experiment/HRNET/0.8/milo_oracle_spat_fronerbius_08_c.yaml

# python -m torch.distributed.launch --nproc_per_node 1 tools/train.py --cfg /home/nickbarry/Documents/MsC-DS/Data_Science_Research_Project/Coresets/Repositories/HRNet-Semantic-Segmentation-Coreset/experiments/pascal_ctx/proxy_experiment/HRNET/0.5/milo_oracle_spat_gromov_wasserstein_05_a.yaml
# python -m torch.distributed.launch --nproc_per_node 1 tools/train.py --cfg /home/nickbarry/Documents/MsC-DS/Data_Science_Research_Project/Coresets/Repositories/HRNet-Semantic-Segmentation-Coreset/experiments/pascal_ctx/proxy_experiment/HRNET/0.5/milo_oracle_spat_gromov_wasserstein_05_b.yaml
# python -m torch.distributed.launch --nproc_per_node 1 tools/train.py --cfg /home/nickbarry/Documents/MsC-DS/Data_Science_Research_Project/Coresets/Repositories/HRNet-Semantic-Segmentation-Coreset/experiments/pascal_ctx/proxy_experiment/HRNET/0.5/milo_oracle_spat_gromov_wasserstein_05_c.yaml
