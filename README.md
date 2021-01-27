# www21
The Implemention of paper "Mask-GVAE: Blind Denoising Graphs via Partition"<sup>[1]</sup>. It is accepted by the WWW 2021: International World Wide Web Conferences .

![Mask-GVAE](https://github.com/halimiqi/www21/blob/master/Mask-GVAE_model.png)  

## Usage

To train the Mask-GVAE model, please run the *main_budget.py* as `python main_budget.py`

The default dataset is PTC_MR with noise edges added. To change the other dataset, please run `python main_budget.py --dataset=[Your dataset index]`. For the modification of other parameters, please visit the main_budget.py.
 
## Environment
The model is implemented based on python=3.6.7 and tensorflow=1.13. Other requirements of the enviorment is listed in *requirements.txt*.

## Setting
The code is training on Nvidia V100 GPU with 16 Gb RAM. The CPU is Intel(R) Xeon(R) Silver 4214R and the memory is 64Gb. This is not the minimum required setting for this project. Other hardware setting may also feasible for this implemention.

This work is collaborated by researchers from the Chinese University of Hong Kong, Georgia Institute of Technology and Huawei Noah's Ark Lab.

---
[1] Li, Jia, et al. "Mask-GVAE: Blind Denoising Graphs via Partition", In Companion Proceedings of the Web Conference 2021 (WWW 2021).
