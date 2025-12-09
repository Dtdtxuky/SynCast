# SynCast: Synergizing Contradictions in Precipitation Nowcasting via Diffusion Sequential Preference Optimization

This repository contains the official PyTorch implementation for the paper "SynCast: Synergizing Contradictions in Precipitation Nowcasting via Diffusion Sequential Preference Optimization" (TCSVT-26462-2025).

## Installation

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/Dtdtxuky/SynCast.git
    cd SynCast
    ```

2.  **Create and Activate a Conda Environment:**
    We recommend using Python 3.8 or higher.
    ```bash
    conda create -n syncast python=3.8
    conda activate syncast
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Dataset Preparation

We are currently in the process of organizing the datasets. We will provide download links and detailed preparation instructions here shortly.

Once available, please download the data and place it in the `./data` directory.

## Training Workflow

The complete training process is divided into three distinct stages. The configuration files (`.yaml`) for each stage are located in the `./configs` directory.

The three stages are:
1.  **Base Model Pre-training:** Training the foundational diffusion model from scratch.
2.  **DPO Fine-tuning for FAR:** Fine-tuning the base model with Direct Preference Optimization (DPO) to reduce the False Alarm Rate (FAR).
3.  **SPO Fine-tuning for CSI:** Further fine-tuning the model from the DPO stage with Sequential Preference Optimization (SPO) to improve the Critical Success Index (CSI).

### How to Run Training

The primary training script is `train.py`. We have provided a convenience script, `train.sh`, to streamline the execution of each stage.

To run a specific training stage, simply execute `train.sh` and pass the path to the corresponding configuration file as an argument.

#### Stage 1: Base Model Pre-training```bash
bash train.sh ./configs/stage1_base_model.yaml
```

#### Stage 2: DPO Fine-tuning for FAR
*Note: This stage requires a pre-trained checkpoint from Stage 1. Ensure the checkpoint path is correctly specified in the config file.*
```bash
bash train.sh ./configs/stage2_dpo_far.yaml
```

#### Stage 3: SPO Fine-tuning for CSI
*Note: This stage requires a fine-tuned checkpoint from Stage 2. Ensure the checkpoint path is correctly specified in the config file.*
```bash
bash train.sh ./configs/stage3_spo_csi.yaml
```

## Citation

If you find our work useful in your research, we would appreciate a citation to our paper.

```bibtex
@article{xu2025syncast,
  title={SynCast: Synergizing Contradictions in Precipitation Nowcasting via Diffusion Sequential Preference Optimization},
  author={Xu, Kaiyi and Gong, Junchao and Zhang, Wenlong and Fei, Ben and Bai, Lei and Ouyang, Wangli},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  year={2025}
}
```

## Acknowledgements

[Optional: Add any acknowledgements here, e.g., to other open-source projects you have used.]
