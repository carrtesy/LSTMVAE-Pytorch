# LSTMVAE-Pytorch

(Unoffical) Implementation of LSTMVAE: [Daehyung Park, Yuuna Hoshi, Charles C. Kemp:
A Multimodal Anomaly Detector for Robot-Assisted Feeding Using an LSTM-Based Variational Autoencoder. IEEE Robotics Autom. Lett. 3(2): 1544-1551 (2018)](https://arxiv.org/pdf/1711.00614.pdf)

If you have noticed errors in implementing, or found better hyperparamters/scores, plz let me know via github issues, pull request, or whatever communication tools you'd prefer.

## Quickstart

```sh
touch secret.py < echo "WANDB_API_KEY={your_wandb_api_key}" # this repo utilizes wandb.
sh run.sh {gpu_id}
```

## Wandb Sweep
```sh
wandb sweep hptune/NTMul.yaml
CUDA_VISIBLE_DEVICES={gpu_id} wandb agent {sweep_id}
```

## Re-implementation so far
F1/F1-PA metrics: 

|                | F1  | F1-PA |
|----------------|-----|-------|
| NeurIPS-TS-MUL | -   | -     |

