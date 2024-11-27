# IRQNet
Implementation of IRQNet.

## Envs
- CUDA 11.7
- PyTorch 1.13
- MMSegmentation 0.30.0

## Quick Start
### Installation
```bash
pip install -U openmim
mim install mmcv-full

pip install -v -e mmsegmentation-0.30.0
```
### Merge the IRQNet folder into the MMSegmentation folder
```bash
cp -r IRQNet/* mmsegmentation-0.30.0/
```

### Prepare dataset
Please refer to "[dataset_prepare.md](https://github.com/nchennnn/IRQNet/blob/main/mmsegmentation-0.30.0/docs/en/dataset_prepare.md)" for dataset preparation.

### Training
```bash
cd mmsegmentation-0.30.0
tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM} [optional arguments]
```
Please refer to "[train.md](https://github.com/nchennnn/IRQNet/blob/main/mmsegmentation-0.30.0/docs/en/train.md)" for training.


## Acknowledgments
IRQNet is built with [MMSegmentation](https://github.com/open-mmlab/mmsegmentation/tree/0.x). Thanks for their awesome work!

```bibtex
@misc{mmseg2020,
    title={{MMSegmentation}: OpenMMLab Semantic Segmentation Toolbox and Benchmark},
    author={MMSegmentation Contributors},
    howpublished = {\url{https://github.com/open-mmlab/mmsegmentation}},
    year={2020}
}
```
