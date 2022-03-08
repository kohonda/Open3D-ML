import os
import sys

# import ml3d
import numpy as np

# ml3dをinclude
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import open3d
# from ml3d.vis import Visualizer
import open3d.ml.torch as ml3d_pip
import scipy.spatial as ss
from ml3d.datasets import SemanticKITTI
from ml3d.torch.models import KPFCNN, RandLANet
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
        # print(features[target_point_idx])
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
    # model = KPFCNN(**cfg.model)
    cfg.dataset['dataset_path'] = "/media/honda/ssd/kitti_data/data_odometry_velodyne"

    dataset = SemanticKITTI(cfg.dataset.pop('dataset_path', None), **cfg.dataset)
    pipeline = SemanticSegmentation(model, dataset=dataset, device="cpu", **cfg.pipeline)

    # download the weights.

    ckpt_path = example_dir + "/vis_weights_{}.pth".format('RandLANet')

    # load the parameters.
    pipeline.load_ckpt(ckpt_path=ckpt_path)
    data_split = dataset.get_split("train")

    # Output setup
    output_root = "/media/honda/ssd/kitti_data/features/randlanet"

    sequence_list  = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]
    # sequence_list  = ["05", "07", "09", "06", "08", "10"]
    
    for sequence in sequence_list:
        print("sequence: ", sequence)

        output_dir = os.path.join(output_root, sequence)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        print("output_dir: ", output_dir)

        data_size = data_split.get_data_size(sequence)
        
        for idx in range(data_size):
            print("sequence; ", sequence , " | progress: ", idx, "/", data_size)

            # output file
            file_name = '{:06d}'.format(idx)
            output_file = os.path.join(output_dir, "{}.txt".format(file_name))
            if os.path.exists(output_file):
                continue
            
            data= data_split.get_data_sequence(sequence, idx)
            result = pipeline.run_inference(data)
            features = result['features']

            pca = PCA(n_components=12)
            compressed_features = pca.fit_transform(features)

            # save features
            np.savetxt(output_file, compressed_features, fmt="%.6f")


        



    

    


