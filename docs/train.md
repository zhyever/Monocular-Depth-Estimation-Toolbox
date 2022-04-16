## Train a model

Monocular-Depth-Estimation-Toolbox implements distributed training and non-distributed training, which uses `MMDistributedDataParallel` and `MMDataParallel` respectively.

All outputs (log files and checkpoints) will be saved to the working directory, which is specified by `work_dir` in the config file.

By default we evaluate the model on the validation set after some epoches, you can change the evaluation interval by adding the interval argument in the training config.

```python
evaluation = dict(interval=2)  # This evaluate the model per 2 epoches.
```

**\*Important\***: The default learning rate in config files is for 2 GPUs and 8 img/gpu (batch size = 2x8 = 16). Equivalently, you may also use 8 GPUs and 2 imgs/gpu since all models using cross-GPU SyncBN.

### Train on a single machine

#### Train with a single GPU

official support:

```shell
sh tools/dist_train.sh ${CONFIG_FILE} 1 [optional arguments]
```

experimental support (you may need to set PYTHONPATH):

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

If you want to specify the working directory in the command, you can add an argument `--work-dir ${YOUR_WORK_DIR}`.

#### Train with CPU

The process of training on the CPU is consistent with single GPU training. We just need to disable GPUs before the training process.

```shell
export CUDA_VISIBLE_DEVICES=-1
```

And then run the script [above](#train-with-a-single-gpu).

```{warning}
The process of training on the CPU is consistent with single GPU training. We just need to disable GPUs before the training process.
```

#### Train with multiple GPUs

```shell
sh tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM} [optional arguments]
```

Optional arguments are:

- `--no-validate` (**not suggested**): By default, the codebase will perform evaluation at every k iterations during the training. To disable this behavior, use `--no-validate`.
- `--work-dir ${WORK_DIR}`: Override the working directory specified in the config file.
- `--resume-from ${CHECKPOINT_FILE}`: Resume from a previous checkpoint file (to continue the training process).
- `--load-from ${CHECKPOINT_FILE}`: Load weights from a checkpoint file (to start finetuning for another task).
- `--deterministic`: Switch on "deterministic" mode which slows down training but the results are reproducible.

Difference between `resume-from` and `load-from`:

- `resume-from` loads both the model weights and optimizer state including the iteration number.
- `load-from` loads only the model weights, starts the training from iteration 0.

An example:

```shell
# checkpoints and logs saved in WORK_DIR=work_dirs/saves/depthformer/depthformer_swint_w7_nyu
# If work_dir is not set, it will be generated automatically.
bash ./tools/dist_train.sh configs/depthformer/depthformer_swint_w7_nyu.py 2 --work-dir work_dirs/saves/depthformer/depthformer_swint_w7_nyu
```

**Note**: During training, checkpoints and logs are saved in the same folder structure as the config file under `work_dirs/`. Custom work directory is not recommended since evaluation scripts infer work directories from the config file name. If you want to save your weights somewhere else, please use symlink, for example:

```shell
ln -s ${YOUR_WORK_DIRS} ${TOOLBOX}/work_dirs
```

#### Launch multiple jobs on a single machine

If you launch multiple jobs on a single machine, e.g., 2 jobs of 4-GPU training on a machine with 8 GPUs, you need to specify different ports (29500 by default) for each job to avoid communication conflict. Otherwise, there will be error message saying `RuntimeError: Address already in use`.

If you use `dist_train.sh` to launch training jobs, you can set the port in commands with environment variable `PORT`.

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 sh tools/dist_train.sh ${CONFIG_FILE} 4
CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=29501 sh tools/dist_train.sh ${CONFIG_FILE} 4
```

### Train with multiple machines

If you launch with multiple machines simply connected with ethernet, you can simply run following commands:

On the first machine:

```shell
NNODES=2 NODE_RANK=0 PORT=$MASTER_PORT MASTER_ADDR=$MASTER_ADDR sh tools/dist_train.sh $CONFIG $GPUS
```

On the second machine:

```shell
NNODES=2 NODE_RANK=1 PORT=$MASTER_PORT MASTER_ADDR=$MASTER_ADDR sh tools/dist_train.sh $CONFIG $GPUS
```

Usually it is slow if you do not have high speed networking like InfiniBand.

### Manage jobs with Slurm

Slurm is a good job scheduling system for computing clusters. On a cluster managed by Slurm, you can use slurm_train.sh to spawn training jobs. It supports both single-node and multi-node training.

Train with multiple machines:

```shell
[GPUS=${GPUS}] sh tools/slurm_train.sh ${PARTITION} ${JOB_NAME} ${CONFIG_FILE} --work-dir ${WORK_DIR}
```

Here is an example of using 16 GPUs to train DepthFormer on the dev partition.

```shell
GPUS=16 sh tools/slurm_train.sh dev depthformer configs/depthformer/depthformer_swint_w7_nyu.py --work-dir work_dirs/saves/depthformer/depthformer_swint_w7_nyu
```

When using 'slurm_train.sh' to start multiple tasks on a node, different ports need to be specified. Three settings are provided:

Option 1:

In `config1.py`:

```python
dist_params = dict(backend='nccl', port=29500)
```

In `config2.py`:

```python
dist_params = dict(backend='nccl', port=29501)
```

Then you can launch two jobs with config1.py and config2.py.

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 GPUS=4 sh tools/slurm_train.sh ${PARTITION} ${JOB_NAME} config1.py tmp_work_dir_1
CUDA_VISIBLE_DEVICES=4,5,6,7 GPUS=4 sh tools/slurm_train.sh ${PARTITION} ${JOB_NAME} config2.py tmp_work_dir_2
```

Option 2:

You can set different communication ports without the need to modify the configuration file, but have to set the `cfg-options` to overwrite the default port in configuration file.

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 GPUS=4 sh tools/slurm_train.sh ${PARTITION} ${JOB_NAME} config1.py tmp_work_dir_1 --cfg-options dist_params.port=29500
CUDA_VISIBLE_DEVICES=4,5,6,7 GPUS=4 sh tools/slurm_train.sh ${PARTITION} ${JOB_NAME} config2.py tmp_work_dir_2 --cfg-options dist_params.port=29501
```

Option 3:

You can set the port in the command using the environment variable 'MASTER_PORT':

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 GPUS=4 MASTER_PORT=29500 sh tools/slurm_train.sh ${PARTITION} ${JOB_NAME} config1.py tmp_work_dir_1
CUDA_VISIBLE_DEVICES=4,5,6,7 GPUS=4 MASTER_PORT=29501 sh tools/slurm_train.sh ${PARTITION} ${JOB_NAME} config2.py tmp_work_dir_2
```

### Runtime Logs
As default, we utilize to `TextLoggerHook` and `TensorboardImageLoggerHook` to log information during training. 

The former prints log in the shell as:
```shell
2022-04-03 00:29:11,300 - depth - INFO - Epoch [3][1200/1514]   lr: 3.543e-05, eta: 3:13:52, time: 0.357, data_time: 0.009, memory: 15394, decode.loss_depth: 0.1381, loss: 0.1381, grad_norm: 1.4511
2022-04-03 00:29:29,139 - depth - INFO - Epoch [3][1250/1514]   lr: 3.608e-05, eta: 3:13:32, time: 0.357, data_time: 0.009, memory: 15394, decode.loss_depth: 0.1420, loss: 0.1420, grad_norm: 1.5763
```

The later saves loss/acc curves and images in the tensorboard server. After start tensorboard and open the page, you can watch the training process.

<div align=center><img src="../resources/tensorboard_loss.png"/></div>
<div align=center><img src="../resources/tensorboard_img.png"/></div>
 