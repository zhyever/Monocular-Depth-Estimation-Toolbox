import mmcv
import os
import numpy as np
import cv2

res_path = 'nfs/results/test-ensemble-res'

path_names = ['nfs/saves/test-ensemble']

weights = [1]

reweights = [w/sum(weights) for w in weights]

file_names = os.listdir(path_names[0])

for name in file_names:
    for idx, (path_name, w) in enumerate(zip(path_names, reweights)):
        file_path = os.path.join(path_name, name)
        if idx == 0:
            temp_res = w * np.load(file_path)
        else:
            temp_res += w * np.load(file_path)
    ensemble_res = temp_res / len(path_names)
    ensemble_res = ensemble_res[0].astype(np.uint16)
    filename = name[:-4]
    filename = filename + '.png'
    mmcv.imwrite(ensemble_res, os.path.join(res_path, filename))