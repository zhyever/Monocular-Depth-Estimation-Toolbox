from logging import raiseExceptions
import os.path as osp
import warnings
from collections import OrderedDict
from functools import reduce

import mmcv
import numpy as np
from mmcv.utils import print_log
from prettytable import PrettyTable
from torch.utils.data import Dataset

from depth.core import pre_eval_to_metrics, metrics, eval_metrics
from depth.utils import get_root_logger
from depth.datasets.builder import DATASETS
from depth.datasets.pipelines import Compose

from depth.ops import resize

from PIL import Image

import torch
import os


@DATASETS.register_module()
class CustomDepthDataset(Dataset):
    """Custom dataset for depth esitmation. An example of file structure.
    is as followed.
    .. code-block:: none
        ├── data
        │   ├── custom
        │   │   ├── img1
        │   │   ├── img2
        │   │   │   ...
        │   │   │   ...

    Args:
        pipeline (list[dict]): Processing pipeline
        img_dir (str): Path to image directory
        data_root (str, optional): Data root for img_dir.
        test_mode (bool): test_mode=True
        min_depth=1e-3: Default min depth value.
        max_depth=10: Default max depth value.
    """

    def __init__(self,
                 pipeline,
                 data_root,
                 test_mode=True,
                 min_depth=1e-3,
                 max_depth=10):

        self.pipeline = Compose(pipeline)
        self.data_root = data_root
        self.test_mode = test_mode
        self.min_depth = min_depth
        self.max_depth = max_depth

        # load annotations
        self.img_infos = self.load_annotations(self.data_root)
        

    def __len__(self):
        """Total number of samples of data."""
        return len(self.img_infos)

    def load_annotations(self, img_dir):
        """Load annotation from directory.
        Args:
            img_dir (str): Path to image directory. Load all the images under the root.
        Returns:
            list[dict]: All image info of dataset.
        """

        self.invalid_depth_num = 0
        img_infos = []

        imgs = os.listdir(img_dir)
        imgs.sort()
        for img_name in imgs:
            img_info = dict()
            img_info['filename'] = img_name
            img_infos.append(img_info)

        # github issue:: make sure the same order
        img_infos = sorted(img_infos, key=lambda x: x['filename'])
        print_log(f'Loaded {len(img_infos)} images. Totally {self.invalid_depth_num} invalid pairs are filtered', logger=get_root_logger())
        return img_infos

    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        results['depth_fields'] = []
        results['img_prefix'] = self.data_root

    def __getitem__(self, idx):
        """Get training/test data after pipeline.
        Args:
            idx (int): Index of data.
        Returns:
            dict: Training/test data (with annotation if `test_mode` is set
                False).
        """
        if self.test_mode:
            return self.prepare_test_img(idx)

    def prepare_test_img(self, idx):
        """Get testing data after pipeline.
        Args:
            idx (int): Index of data.
        Returns:
            dict: Testing data after pipeline with new keys introduced by
                pipeline.
        """

        img_info = self.img_infos[idx]
        results = dict(img_info=img_info)
        self.pre_pipeline(results)
        return self.pipeline(results)

    def format_results(self, results, imgfile_prefix=None, indices=None, **kwargs):
        """Place holder to format result to dataset specific output."""
        results[0] = (results[0] * self.depth_scale) # Do not convert to np.uint16 for ensembling. # .astype(np.uint16)
        return results

    def pre_eval(self, preds, indices):
        pass

    def evaluate(self, results, metric='eigen', logger=None, **kwargs):
        pass
