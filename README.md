# SynCast: Synergizing Contradictions in Precipitation Nowcasting via Diffusion Sequential Preference Optimization

This repository contains the official PyTorch implementation for the paper "SynCast: Synergizing Contradictions in Precipitation Nowcasting via Diffusion Sequential Preference Optimization".

## Installation

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/Dtdtxuky/SynCast.git
    cd SynCast
    ```

2.  **Create and Activate a Conda Environment:**
    We recommend using Python 3.10 or higher.
    ```bash
    conda create -n syncast python=3.8
    conda activate syncast
    ```

## Training Workflow

The complete training process is divided into three distinct stages. The configuration files (`.yaml`) for each stage are located in the `./configs` directory.

The three stages are:
1.  **Base Model Pre-training:** Training the foundational diffusion model from scratch.
2.  **DPO Fine-tuning for FAR:** Fine-tuning the base model with Direct Preference Optimization (DPO) to reduce the False Alarm Rate (FAR).
3.  **SPO Fine-tuning for CSI:** Further fine-tuning the model from the DPO stage with Sequential Preference Optimization (SPO) to improve the Critical Success Index (CSI).

### How to Run Training

The primary training script is `train.py`. We have provided a convenience script, `train.sh`, to streamline the execution of each stage.

To run a specific training stage, simply execute `train.sh` and pass the path to the corresponding configuration file as an argument.


## Citation

If you find our work useful in your research, we would appreciate a citation to our paper.

```bibtex
@article{xu2025syncast,
  title={SynCast: Synergizing Contradictions in Precipitation Nowcasting via Diffusion Sequential Preference Optimization},
  author={Xu, Kaiyi and Gong, Junchao and Zhang, Wenlong and Fei, Ben and Bai, Lei and Ouyang, Wanli},
  journal={arXiv preprint arXiv:2510.21847},
  year={2025}
}
```

