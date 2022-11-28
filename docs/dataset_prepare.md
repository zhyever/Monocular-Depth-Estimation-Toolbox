## Prepare datasets

It is recommended to symlink the dataset root to `$MONOCULAR-DEPTH-ESTIMATION-TOOLBOX/data`.
If your folder structure is different, you may need to change the corresponding paths in config files.

```none
monocular-depth-estimation-toolbox
├── depth
├── tools
├── configs
├── splits
├── data
│   ├── kitti
│   │   ├── input
│   │   │   ├── 2011_09_26
│   │   │   ├── 2011_09_28
│   │   │   ├── ...
│   │   │   ├── 2011_10_03
│   │   ├── gt_depth
│   │   │   ├── 2011_09_26_drive_0001_sync
│   │   │   ├── 2011_09_26_drive_0002_sync
│   │   │   ├── ...
│   │   │   ├── 2011_10_03_drive_0047_sync
|   |   ├── benchmark_test
│   │   │   ├── 0000000000.png
│   │   │   ├── 0000000001.png
│   │   │   ├── ...
│   │   │   ├── 0000000499.png
|   |   ├── benchmark_cam
│   │   │   ├── 0000000000.txt
│   │   │   ├── 0000000001.txt
│   │   │   ├── ...
│   │   │   ├── 0000000499.txt
│   │   ├── split_file.txt
│   ├── nyu
│   │   ├── basement_0001a
│   │   ├── basement_0001b
│   │   ├── ... (all scene names)
│   │   ├── split_file.txt
│   ├── SUNRGBD
│   │   ├── SUNRGBD
│   │   │   ├── kv1
│   │   │   ├── kv2
│   │   │   ├── realsense
│   │   │   ├── xtion
│   │   ├── split_file.txt
│   ├── cityscapes
│   │   ├── camera
│   │   │   ├── test
│   │   │   ├── train
│   │   │   ├── val
│   │   ├── disparity_trainvaltest
│   │   │   ├── disparity
│   │   ├── leftImg8bit_trainvaltest
│   │   │   ├── leftImg8bit
│   │   ├── split_file.txt
│   ├── cityscapesExtra
│   │   ├── camera
│   │   │   ├── train_extra
│   │   ├── disparity
│   │   │   ├── train_extra
│   │   ├── leftImg8bit
│   │   │   ├── train_extra
│   │   ├── split_file.txt
│   ├── custom_dataset
│   │   ├── train
│   │   │   ├── rgb
│   │   │   │   ├── 0.xxx
│   │   │   │   ├── 1.xxx
│   │   │   │   ├── 2.xxx
│   │   │   ├── depth
│   │   │   │   ├── 0.xxx
│   │   │   │   ├── 1.xxx
│   │   │   │   ├── 2.xxx
│   │   ├── val
│   │   │   ├── rgb
│   │   │   │   ├── 0.xxx
│   │   │   │   ├── 1.xxx
│   │   │   │   ├── 2.xxx
│   │   │   ├── depth
│   │   │   │   ├── 0.xxx
│   │   │   │   ├── 1.xxx
│   │   │   │   ├── 2.xxx
```

### **KITTI**

Download the offical dataset from this [link](http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction), including the raw data (about 200G) and fine-grained ground-truth depth maps. 

Then, unzip the files into data/kitti. Remember to organizing the directory structure following instructions (Only need a few cut operations). 

Finally, copy split files (whose names are started with *kitti*) in splits folder into data/kitti. Here, I utilize eigen splits following other supervised methods.

Some methods may use the camera intrinsic parameters (*i.e.,* BTS), you need to download the [benchmark_cam](https://drive.google.com/file/d/1ktSDTUx9dDViBKoAeqTERTay1813xfUK/view?usp=sharing) consisting of camera intrinsic parameters of the benchmark test set.

### **NYU**

Following previous work, I utilize about 24231 image-depth pairs as the training set and standard 652 images as the validation set. You can download the dataset from [Google Drive Link](https://drive.google.com/file/d/1wC-io-14RCIL4XTUrQLk6lBqU2AexLVp/view?usp=share_link) (including training and validation sets).

<!-- You can download the subset with the help of codes provided in [BTS](https://github.com/cleinc/bts/tree/master/pytorch).

```shell
$ git clone https://github.com/cleinc/bts.git
$ cd bts
$ python utils/download_from_gdrive.py 1AysroWpfISmm-yRFGBgFTrLy6FjQwvwP sync.zip
$ unzip sync.zip
```

Also, you can download it from following link: https://drive.google.com/file/d/1AysroWpfISmm-yRFGBgFTrLy6FjQwvwP/view?usp=sharing

Then, you need to download the standard test set from this [link](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html). (**Note**: The downloaded file will be unzipped to folder test and train. You need to cut the files in the test folder out to data/nyu, organizing the directory structure following the file trees provided on the top of this page.)

Finally, copy nyu_train.txt and nyu_test.txt in the splits folder into the data/nyu. -->


### **SUNRGBD**

The dataset could be download from this [link](https://rgbd.cs.princeton.edu/). Copy SUNRGBD_val_splits.txt in splits into data/SUNRGBD.

### **Cityscapes and CityscapesExtra**

The data could be found [here](https://www.cityscapes-dataset.com/downloads/) after registration. Copy cityscapes_train.txt and cityscapes_train in splits to data/cityscapes. If using extra data, copy cityscapes_train_extra.txt to data/cityscapesExtra.


### **Custom Dataset**

We also provide a simple custom dataset class for users in `depth/datasets/custom.py`. Organize your data folder as our illustration. Note that instead of utilizing a split file to divide the train/val set, we directly classify data into train/val folder. A simple config file can be like:

```
train=dict(
    type=dataset_type,
    pipeline=dict(...),
    data_root='data/custom_dataset',
    test_mode=False,
    min_depth=1e-3,
    max_depth=10,
    depth_scale=1)
```

As for the custom dataset, we do not implement the evaluation details. If you want to get a quantitive metric result, you need to implement the `pre_eval` and `evaluate` functions following the ones in KITTI or other datasets.