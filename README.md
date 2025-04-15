# FACT: Foundation model for Assessing Cancer Tissue margins

![Graphical Abstract](.figures/graphical-abstract.png)

PyTorch implementation of [FACT: Foundation model for Assessing Cancer Tissue margins](#).

## Setup

Start by cloning the repository:

```bash
git clone https://github.com/med-i-lab/fact.git
```

and then installing PyTorch v2.3 in a new virtual environment. Follow the instructions on the [official website](https://pytorch.org/get-started/previous-versions/#v231) to install PyTorch.

Finally, navigate to the project directory and install the other dependencies:

```bash
pip install -r requirements.txt
```

## Pretraining

To pretrain the model, run the following command:

```bash
python train.py with Triplet base_model=laion/clap-htsat-unfused dataset=<PATH>
```

This will load in the configuration file, `configs/Triplet.yaml`, and start pretraining the model on the dataset located at `<PATH>`. The configuration file contains all the hyperparameters for the model, optimizer, and training loop. You can override the default values, e.g., the optimizer and learning rate, by passing them as arguments:

```
python train.py with Triplet base_model=laion/clap-htsat-unfused dataset=<PATH> optimizer=SGD lr=0.01
```

See `configs/defaults.yaml` for all the available hyperparameters.

### Backbones

There are three backbones available for pretraining: `laion/clap-htsat-unfused`, `openai/clip-vit-base-patch32`, and `pluskal-lab/DreaMS`. You can specify the backbone by passing the `base_model` argument. For CLIP, you also need to specify `transform` (and `transform_params`). To simplify the process, we have provided an additional configuration file, `configs/CLIP.yaml`, that contains the default values for CLIP:

```
python train.py with Triplet CLIP dataset=<PATH>
```

### Ablation Studies

The configuration files for FixMatch and SimCLR are also available in the `configs` directory. You can reproduce the ablation studies using the provided values.

## Finetuning

To finetune the model, run the following command:

```bash
python train.py with base_model=laion/clap-htsat-unfused dataset=<PATH> checkpoint=<CHECKPOINT>
```

Note that you do not need to specify the `Triplet` configuration since the model has already been pretrained. Instead, you need to reference the final (or best) checkpoint file from the pretraining step.

## Evaluation

Model checkpoints (all 30 finetuning runs, and top 10 pretraining runs) are available at `/mnt/mirage/med-i_data/Data/checkpoints/FACT_IPCAI2025`. To reproduce the results of the paper, run the following command:

```bash
python train.py with base_model=laion/clap-htsat-unfused dataset=<PATH> checkpoint=<CHECKPOINT> eval_only=True
````

## Specs

For our experiments, we used an NVIDIA Quadro RTX 6000 (24GB) GPU and 32GB of RAM. With the default hyperparameters, the model should fit in the GPU memory. If you encounter any issues, consider reducing the batch size.

## Citation

> Farahmand, M., Jamzad, A., Fooladgar, F., Connolly, L., Kaufmann, M., Ren, K.Y.M., Rudan, J., McKay, D., Fichtinger, G. and Mousavi, P., 2025. "FACT: Foundation Model for Assessing Cancer Tissue Margins with Mass Spectrometry". International Journal of Computer Assisted Radiology and Surgery.

```bibtex
﻿@article{Farahmand2025,
author={Farahmand, Mohammad
and Jamzad, Amoon
and Fooladgar, Fahimeh
and Connolly, Laura
and Kaufmann, Martin
and Ren, Kevin Yi Mi
and Rudan, John
and McKay, Doug
and Fichtinger, Gabor
and Mousavi, Parvin},
title={FACT: Foundation Model for Assessing Cancer Tissue Margins with Mass Spectrometry},
journal={International Journal of Computer Assisted Radiology and Surgery},
year={2025},
month={Apr},
day={04},
issn={1861-6429},
doi={10.1007/s11548-025-03355-8},
url={https://doi.org/10.1007/s11548-025-03355-8}
}
```

