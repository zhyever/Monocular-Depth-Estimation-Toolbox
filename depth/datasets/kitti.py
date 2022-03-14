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


@DATASETS.register_module()
class KITTIDataset(Dataset):
    """KITTI dataset for depth estimation. An example of file structure
    is as followed.
    .. code-block:: none
        ├── data
        │   ├── KITTI
        │   │   ├── kitti_eigen_train.txt
        │   │   ├── kitti_eigen_test.txt
        │   │   ├── input (RGB, img_dir)
        │   │   │   ├── date_1
        │   │   │   ├── date_2
        │   │   │   |   ...
        │   │   │   |   ...
        |   │   ├── gt_depth (ann_dir)
        │   │   │   ├── date_drive_number_sync
    split file format:
    input_image: 2011_09_26/2011_09_26_drive_0002_sync/image_02/data/0000000069.png 
    gt_depth:    2011_09_26_drive_0002_sync/proj_depth/groundtruth/image_02/0000000069.png 
    focal:       721.5377 (following the focal setting in BTS, but actually we do not use it)
    Args:
        pipeline (list[dict]): Processing pipeline
        img_dir (str): Path to image directory
        ann_dir (str, optional): Path to annotation directory. Default: None
        split (str, optional): Split txt file. Split should be specified, only file in the splits will be loaded.
        data_root (str, optional): Data root for img_dir/ann_dir. Default: None.
        test_mode (bool): If test_mode=True, gt wouldn't be loaded.
        depth_scale=256: Default KITTI pre-process. divide 256 to get gt measured in meters (m)
        garg_crop=True: Following Adabins, use grag crop to eval results.
        eigen_crop=False: Another cropping setting.
        min_depth=1e-3: Default min depth value.
        max_depth=80: Default max depth value.
    """


    def __init__(self,
                 pipeline,
                 img_dir,
                 ann_dir=None,
                 split=None,
                 data_root=None,
                 test_mode=False,
                 depth_scale=256,
                 garg_crop=True,
                 eigen_crop=False,
                 min_depth=1e-3,
                 max_depth=80):

        self.pipeline = Compose(pipeline)
        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.split = split
        self.data_root = data_root
        self.test_mode = test_mode
        self.depth_scale = depth_scale
        self.garg_crop = garg_crop
        self.eigen_crop = eigen_crop
        self.min_depth = min_depth # just for evaluate. (crop gt to certain range)
        self.max_depth = max_depth # just for evaluate.

        # join paths if data_root is specified
        if self.data_root is not None:
            if not (self.img_dir is None or osp.isabs(self.img_dir)):
                self.img_dir = osp.join(self.data_root, self.img_dir)
            if not (self.ann_dir is None or osp.isabs(self.ann_dir)):
                self.ann_dir = osp.join(self.data_root, self.ann_dir)
            if not (self.split is None or osp.isabs(self.split)):
                self.split = osp.join(self.data_root, self.split)

        # load annotations
        self.img_infos = self.load_annotations(self.img_dir, self.ann_dir, self.split)
        

    def __len__(self):
        """Total number of samples of data."""
        return len(self.img_infos)

    def load_annotations(self, img_dir, ann_dir, split):
        """Load annotation from directory.
        Args:
            img_dir (str): Path to image directory
            ann_dir (str|None): Path to annotation directory.
            split (str|None): Split txt file. Split should be specified, only file in the splits will be loaded.
        Returns:
            list[dict]: All image info of dataset.
        """

        self.invalid_depth_num = 0
        img_infos = []
        if split is not None:
            with open(split) as f:
                for line in f:
                    img_info = dict()
                    if ann_dir is not None: # benchmark test or unsupervised future
                        depth_map = line.strip().split(" ")[1]
                        if depth_map == 'None':
                            self.invalid_depth_num += 1
                            continue
                        img_info['ann'] = dict(depth_map=depth_map)
                    img_name = line.strip().split(" ")[0]
                    img_info['filename'] = img_name
                    img_infos.append(img_info)
        else:
            print("Split should be specified, NotImplementedError")
            raise NotImplementedError 

        # github issue:: make sure the same order
        img_infos = sorted(img_infos, key=lambda x: x['filename'])
        print_log(f'Loaded {len(img_infos)} images. Totally {self.invalid_depth_num} invalid pairs are filtered', logger=get_root_logger())

        return img_infos

    def get_ann_info(self, idx):
        """Get annotation by index.
        Args:
            idx (int): Index of data.
        Returns:
            dict: Annotation info of specified index.
        """

        return self.img_infos[idx]['ann']

    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        results['depth_fields'] = []
        results['img_prefix'] = self.img_dir
        results['depth_prefix'] = self.ann_dir
        results['depth_scale'] = self.depth_scale

        results['cam_intrinsic_dict'] = {
            '2011_09_26' : [[7.215377e+02, 0.000000e+00, 6.095593e+02, 4.485728e+01], 
                            [0.000000e+00, 7.215377e+02, 1.728540e+02, 2.163791e-01],
                            [0.000000e+00, 0.000000e+00, 1.000000e+00, 2.745884e-03]],
            '2011_09_28' : [[7.070493e+02, 0.000000e+00, 6.040814e+02, 4.575831e+01], 
                            [0.000000e+00, 7.070493e+02, 1.805066e+02, -3.454157e-01], 
                            [0.000000e+00, 0.000000e+00, 1.000000e+00, 4.981016e-03]],
            '2011_09_29' : [[7.183351e+02, 0.000000e+00, 6.003891e+02, 4.450382e+01], 
                            [0.000000e+00, 7.183351e+02, 1.815122e+02, -5.951107e-01],
                            [0.000000e+00, 0.000000e+00, 1.000000e+00, 2.616315e-03]],
            '2011_09_30' : [[7.070912e+02, 0.000000e+00, 6.018873e+02, 4.688783e+01], 
                            [0.000000e+00, 7.070912e+02, 1.831104e+02, 1.178601e-01], 
                            [0.000000e+00, 0.000000e+00, 1.000000e+00, 6.203223e-03]],
            '2011_10_03' : [[7.188560e+02, 0.000000e+00, 6.071928e+02, 4.538225e+01], 
                            [0.000000e+00, 7.188560e+02, 1.852157e+02, -1.130887e-01], 
                            [0.000000e+00, 0.000000e+00, 1.000000e+00, 3.779761e-03]],
        }

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
        else:
            return self.prepare_train_img(idx)

    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.
        Args:
            idx (int): Index of data.
        Returns:
            dict: Training data and annotation after pipeline with new keys
                introduced by pipeline.
        """

        img_info = self.img_infos[idx]
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info)
        self.pre_pipeline(results)
        return self.pipeline(results)

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
        results[0] = (results[0] * self.depth_scale).astype(np.uint16)
        return results

    def get_gt_depth_maps(self):
        """Get ground truth depth maps for evaluation."""
        for img_info in self.img_infos:
            depth_map = osp.join(self.ann_dir, img_info['ann']['depth_map'])
            depth_map_gt = np.asarray(Image.open(depth_map), dtype=np.float32) / self.depth_scale
            yield depth_map_gt
    
    def eval_kb_crop(self, depth_gt):
        """Following Adabins, Do kb crop for testing"""
        height = depth_gt.shape[0]
        width = depth_gt.shape[1]
        top_margin = int(height - 352)
        left_margin = int((width - 1216) / 2)
        depth_cropped = depth_gt[top_margin: top_margin + 352, left_margin: left_margin + 1216]
        depth_cropped = np.expand_dims(depth_cropped, axis=0)
        return depth_cropped

    def eval_mask(self, depth_gt):
        """Following Adabins, Do grag_crop or eigen_crop for testing"""
        depth_gt = np.squeeze(depth_gt)
        valid_mask = np.logical_and(depth_gt > self.min_depth, depth_gt < self.max_depth)
        if self.garg_crop or self.eigen_crop:
            gt_height, gt_width = depth_gt.shape
            eval_mask = np.zeros(valid_mask.shape)

            if self.garg_crop:
                eval_mask[int(0.40810811 * gt_height):int(0.99189189 * gt_height),
                          int(0.03594771 * gt_width):int(0.96405229 * gt_width)] = 1

            elif self.eigen_crop:
                eval_mask[int(0.3324324 * gt_height):int(0.91351351 * gt_height),
                          int(0.0359477 * gt_width):int(0.96405229 * gt_width)] = 1
        valid_mask = np.logical_and(valid_mask, eval_mask)
        valid_mask = np.expand_dims(valid_mask, axis=0)
        return valid_mask

    def pre_eval(self, preds, indices):
        """Collect eval result from each iteration.
        Args:
            preds (list[torch.Tensor] | torch.Tensor): the depth estimation.
            indices (list[int] | int): the prediction related ground truth
                indices.
        Returns:
            list[torch.Tensor]: (area_intersect, area_union, area_prediction,
                area_ground_truth).
        """
        # In order to compat with batch inference
        if not isinstance(indices, list):
            indices = [indices]
        if not isinstance(preds, list):
            preds = [preds]

        pre_eval_results = []
        pre_eval_preds = []

        for i, (pred, index) in enumerate(zip(preds, indices)):
            depth_map = osp.join(self.ann_dir,
                               self.img_infos[index]['ann']['depth_map'])

            depth_map_gt = np.asarray(Image.open(depth_map), dtype=np.float32) / self.depth_scale
            depth_map_gt = self.eval_kb_crop(depth_map_gt)
            valid_mask = self.eval_mask(depth_map_gt)
            
            eval = metrics(depth_map_gt[valid_mask], 
                           pred[valid_mask], 
                           min_depth=self.min_depth,
                           max_depth=self.max_depth)

            pre_eval_results.append(eval)

            # save prediction results
            pre_eval_preds.append(pred)

        return pre_eval_results, pre_eval_preds

    def evaluate(self, results, metric='eigen', logger=None, **kwargs):
        """Evaluate the dataset.
        Args:
            results (list[tuple[torch.Tensor]] | list[str]): per image pre_eval
                 results or predict depth map for computing evaluation
                 metric.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
        Returns:
            dict[str, float]: Default metrics.
        """
        metric = ["a1", "a2", "a3", "abs_rel", "rmse", "log_10", "rmse_log", "silog", "sq_rel"]
        
        eval_results = {}
        # test a list of files
        if mmcv.is_list_of(results, np.ndarray) or mmcv.is_list_of(
                results, str):
            gt_depth_maps = self.get_gt_depth_maps()
            ret_metrics = eval_metrics(
                gt_depth_maps,
                results)
        # test a list of pre_eval_results
        else:
            ret_metrics = pre_eval_to_metrics(results)
        
        ret_metric_names = []
        ret_metric_values = []
        for ret_metric, ret_metric_value in ret_metrics.items():
            ret_metric_names.append(ret_metric)
            ret_metric_values.append(ret_metric_value)

        num_table = len(ret_metrics) // 9
        for i in range(num_table):
            names = ret_metric_names[i*9: i*9 + 9]
            values = ret_metric_values[i*9: i*9 + 9]

            # summary table
            ret_metrics_summary = OrderedDict({
                ret_metric: np.round(np.nanmean(ret_metric_value), 4)
                for ret_metric, ret_metric_value in zip(names, values)
            })

            # for logger
            summary_table_data = PrettyTable()
            for key, val in ret_metrics_summary.items():
                summary_table_data.add_column(key, [val])

            print_log('Summary:', logger)
            print_log('\n' + summary_table_data.get_string(), logger=logger)

        # each metric dict
        for key, value in ret_metrics.items():
            eval_results[key] = value

        return eval_results
