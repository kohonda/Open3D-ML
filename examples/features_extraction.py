import os
import sys

# import ml3d
import numpy as np

# ml3dをinclude
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import open3d
# from ml3d.vis import Visualizer
import open3d.ml.torch as ml3d_pip
import scipy.spatial as ss
from ml3d.datasets import SemanticKITTI
from ml3d.torch.models import RandLANet
from ml3d.torch.pipelines import SemanticSegmentation
from ml3d.utils.config import Config
from sklearn.decomposition import PCA


def gen_random_color():
    color = np.random.rand(3)
    return color


def coloring_similar_feature_points(points, features, target_point_idx_list, coloring_points_num):
    #  coloring by feature points
    points_colors = np.zeros((len(points), 3))
    
    for target_point_idx in target_point_idx_list:
        tree = ss.KDTree(features)
        _, index = tree.query(features[target_point_idx], coloring_points_num)
    
        color = gen_random_color()
        for idx in index:
            points_colors[idx] = color
   
    return points_colors

def coloring_same_label_points(points, labels, target_point_idx_list):
    #  coloring by feature points
    points_colors = np.zeros((len(points), 3))

    for target_point_idx in target_point_idx_list:
        color = gen_random_color()
        for idx in range(len(points)):
            if labels[target_point_idx] == labels[idx]:
                points_colors[idx] = color

    return points_colors

def pick_points(pcd):
    print("")
    print(
        "1) Please pick point by [shift + left click]"
    )
    print("   Press [shift + right click] to undo point picking")
    print("2) Afther picking points, press q for close the window")
    vis = open3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()  # user picks points
    vis.destroy_window()
    print("")
    return vis.get_picked_points()

if __name__ == "__main__":
    cfg_file = "ml3d/configs/randlanet_semantickitti.yml"
    cfg = Config.load_from_file(cfg_file)
    example_dir = os.path.dirname(os.path.realpath(__file__))

    model = RandLANet(**cfg.model)
    cfg.dataset['dataset_path'] = "/media/honda/ssd/kitti_data/data_odometry_velodyne"

    dataset = SemanticKITTI(cfg.dataset.pop('dataset_path', None), **cfg.dataset)
    pipeline = SemanticSegmentation(model, dataset=dataset, device="cpu", **cfg.pipeline)

    # download the weights.

    ckpt_path = example_dir + "/vis_weights_{}.pth".format('RandLANet')


    # load the parameters.
    pipeline.load_ckpt(ckpt_path=ckpt_path)

    data_split = dataset.get_split("train")

    data = data_split.get_data(10)
    pipeline.model.set_raw_points_num(data['point'].shape[0])
    print("data size: ", data['point'].shape)

    result = pipeline.run_inference(data)
    # torch to numpy
    features = result['features'].cpu().numpy()
    # 2要素目と3要素目を入れ替える
    features = np.swapaxes(features, 1, 2)
    # 2要素目と3要素目を抽出
    features = features[0]
    # [N, 32, 1] -> [N, 32]
    features = features[:, :, 0]
    print("features size: ", features.shape)

    pred_label = (result['predict_labels'] + 1).astype(np.int32)
    # Fill "unlabeled" value because predictions have no 0 values.
    pred_label[0] = 0
    print("label size: ", pred_label.shape)


    # visualize
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(data['point'])
    points_colors = np.zeros((len(data['point']), 3)) # Black
    pcd.colors = open3d.utility.Vector3dVector(points_colors)

    picked_points_index =  pick_points(pcd)
    print("picked points: ", picked_points_index)

    # visualize colored by features
    nearest_points_num = 100
    points_colors = coloring_similar_feature_points(data['point'], features , picked_points_index, nearest_points_num)
    
    pcd.colors = open3d.utility.Vector3dVector(points_colors)
    open3d.visualization.draw_geometries([pcd])

    # visualize colored by labels
    points_colors = coloring_same_label_points(data['point'], pred_label, picked_points_index)
    pcd.colors = open3d.utility.Vector3dVector(points_colors)
    open3d.visualization.draw_geometries([pcd])


