# SAFA: Scale-Adaptive Feature Aggregation for Efficient Space-Time Video Super-Resolution

## Overview

This repository implements the **Scale-Adaptive Feature Aggregation (SAFA)** network for trainable design. (The learnable code is now public [here](https://github.com/hzwer/WACV2024-SAFA/blob/main/train.py))

## Installation

Clone the repository and install the necessary dependencies:

```bash
git clone https://github.com/kmss1258/MS-STVFI
cd MS-STVFI
pip install -r requirements.txt
```

## Training
```bash
python3 -m torch.distributed.run --nproc_per_node=1 train_refactoring.py --world_size=1
```

## Data Prepare

``` 
wget http://data.csail.mit.edu/tofu/dataset/vimeo_triplet.zip
unzip -q vimeo_triplet.zip -d /where/you/want/directory
# use sequences folder data, and reference to train.txt, test.txt
```
