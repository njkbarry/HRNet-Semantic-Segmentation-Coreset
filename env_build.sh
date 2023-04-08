#!/bin/bash

module load fosscuda/2020b
module load pytorch/1.7.1-python-3.8.6
module load opencv/4.5.1-python-3.8.6-contrib
module load pyyaml/5.3.1-python-3.8.6
moduel load cython/0.29.22
module load ninja/1.10.1-python-3.8.6
module load tqdm/4.60.0

virtualenv venv
source venv/bin/activate

python -m install -r requirements_spartan.txt

deactivate
