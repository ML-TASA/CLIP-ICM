# CLIP-ICM
![Static Badge](https://img.shields.io/badge/ICML25-yellow)
![Static Badge](https://img.shields.io/badge/to_be_continue-orange)
![Stars](https://img.shields.io/github/stars/ML-TASA/CLIP-ICM)

This repository provides Pytorch implementation for [ICML2025] Learning Invariant Causal Mechanism from Vision-Language Models.

<p align="center">
<img src=".\figs\figure3.png" height = "320" alt="" align=center />
<br><br>
<b>Figure 1.</b> Overview of CLIP-ICM.
</p>


### Quick Start

```bash
# create env
conda create -n clip-icm python=3.9 -y
conda activate clip-icm

# install deps
pip install -r requirements.txt          
````


### Directory Layout

```text
├── CLIP/               # CLIP model implementation and related files
├── DomainBed/          # Domain generalization benchmark
├── clip_icm.py         # CLIP ICM-related functionality
├── converter_domainbed.py # DomainBed data conversion utilities
├── engine.py           # Training engine
├── imagenet_stubs.py   # ImageNet stubs for testing
├── main.py             # Main entry point for the project
├── README.md           # Project-level README
├── requirements.txt    # Python dependencies for the project
├── utils.py            # Utility functions
```



### Citation

If you find our work and codes useful, please consider citing our paper and star our repository (🥰🎉Thanks!!!):

```bibtex
@inproceedings{songLearningInvariantCausal2025,
  title = {Learning {{Invariant Causal Mechanism}} from {{Vision-Language Models}}},
  booktitle = {Forty-Second {{International Conference}} on {{Machine Learning}}},
  author = {Song, Zeen and Zhao, Siyu and Zhang, Xingyu and Li, Jiangmeng and Zheng, Changwen and Qiang, Wenwen},
  year = {2025},
  month = may,
  urldate = {2025-06-06},
  langid = {english}
}
```


