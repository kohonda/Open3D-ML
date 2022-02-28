import os
import sys

# import ml3d
import numpy as np

# ml3dをinclude
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from ml3d.datasets import SemanticKITTI
from ml3d.torch.models import RandLANet
from ml3d.torch.pipelines import SemanticSegmentation
from ml3d.utils.config import Config

cfg_file = "ml3d/configs/randlanet_semantickitti.yml"
cfg = Config.load_from_file(cfg_file)
example_dir = os.path.dirname(os.path.realpath(__file__))

model = RandLANet(**cfg.model)
data_path = os.path.join(example_dir, 'demo_data')
cfg.dataset['dataset_path'] = "/media/honda/ssd/kitti_data/data_odometry_velodyne"

dataset = SemanticKITTI(cfg.dataset.pop('dataset_path', None), **cfg.dataset)
pipeline = SemanticSegmentation(model, dataset=dataset, device="cpu", **cfg.pipeline)

# download the weights.

ckpt_path = example_dir + "/vis_weights_{}.pth".format('RandLANet')


# load the parameters.
pipeline.load_ckpt(ckpt_path=ckpt_path)

data_split = dataset.get_split("train")
data = data_split.get_data(0)

# print(data['point'].shape)

# run inference on a single example.
# returns dict with 'predict_labels' and 'predict_scores'.
result = pipeline.run_inference(data)

# evaluate performance on the test set; this will write logs to './logs'.
# pipeline.run_test()

print(result.keys())
