# Benchmark and Model Zoo

## Common settings

* We use distributed training with 2 GPUs by default. For different settings such as transformer backbones, we will illustrate in the benchmark.
* (TODO) For the consistency across different hardwares, we report the GPU memory as the maximum value of `torch.cuda.max_memory_allocated()` for all 4 GPUs with `torch.backends.cudnn.benchmark=False`.
  Note that this value is usually less than what `nvidia-smi` shows.
* (TODO) We report the inference time as the total time of network forwarding and post-processing, excluding the data loading time.
  Results are obtained with the script `tools/benchmark.py` which computes the average time on 200 images with `torch.backends.cudnn.benchmark=False`.
* **(TODO)** For input size of 8x+1 (e.g. 769), `align_corner=True` is adopted as a traditional practice.
  Otherwise, for input size of 8x (e.g. 512, 1024), `align_corner=False` is adopted.
  I think there are potential discrepancies here. Take an instance of Adabins, will the input is not 8x+1, it uses `align_corner=True` in their offical implementation. The influence to results is not proved. More exps TBD.

## Baselines

### BTS

Please refer to [BTS](https://github.com/zhyever/Monocular-Depth-Estimation-Toolbox/blob/main/configs/bts) for details.

### Adabins

Please refer to [Adabins](https://github.com/zhyever/Monocular-Depth-Estimation-Toolbox/blob/main/configs/adabins) for details.

### DPT

*This is a simple implementation. Only model structure is aligned with original paper. More experiments about training settings or loss functions are needed to be done.*

Please refer to [DPT](https://github.com/zhyever/Monocular-Depth-Estimation-Toolbox/blob/main/configs/dpt) for details. 


### SimIPU

Please refer to [SimIPU](https://github.com/zhyever/Monocular-Depth-Estimation-Toolbox/blob/main/configs/simipu) for details.

### DepthFormer

Please refer to [DepthFormer](https://github.com/zhyever/Monocular-Depth-Estimation-Toolbox/blob/main/configs/depthformer) for details.

