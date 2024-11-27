# IRQNet

Implementation of IRQNet. Please read [the document](https://mmsegmentation.readthedocs.io/en/latest/) first for using mmsegmentation.

- ir_head.py: main network structure.
- ir_hook.py: mmseg-style hook.
- hr18_ir.py, irqnet_hr32_512x1024_80k_cityscapes.py, irqnet_hr32_512x512_160k_ade20k.py: mmseg-style config.

## Envs

- CUDA11.7
- Pytorch 1.13
- mmsegmentation 0.30

## Citation

Refer to the following, thanks!

```bibtex
@misc{mmseg2020,
    title={{MMSegmentation}: OpenMMLab Semantic Segmentation Toolbox and Benchmark},
    author={MMSegmentation Contributors},
    howpublished = {\url{https://github.com/open-mmlab/mmsegmentation}},
    year={2020}
}
```
