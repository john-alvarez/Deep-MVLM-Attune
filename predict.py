import argparse
from parse_config import ConfigParser
import deepmvlm
from utils3d import Utils3D
import os
import numpy as np
from pathlib import Path

def process_one_file(config, mesh_path):

    from pathlib import Path

    # Example usage
    model_subpath = "Attune Machine Learning Data/Trained Models/MVLMModel_DTU3D_Depth_19092019_only_state_dict-95b89b63.pth"
    model_path = get_mesh_path(mesh_path, model_subpath)

    print('Processing ', mesh_path)
    name_lm_vtk = os.path.splitext(mesh_path)[0] + '_landmarks.vtk'
    name_lm_txt = os.path.splitext(mesh_path)[0] + '_landmarks.txt'
    dm = deepmvlm.DeepMVLM(config, model_path)
    landmarks, heatmap_maxima = dm.predict_one_file(mesh_path)
    print(heatmap_maxima)
    probabilities = heatmap_maxima[:,:,2]
    print(probabilities)
    dm.write_landmarks_as_vtk_points(landmarks, name_lm_vtk)
    dm.write_landmarks_as_text(landmarks, name_lm_txt)
#     dm.visualise_mesh_and_landmarks(mesh_path, landmarks)

    path, basename = os.path.split(mesh_path)
    basename_heatmap = basename.replace("mesh_normalized.obj", "heatmap_probabilities.txt")


    full_path_heatmap = os.path.join(path, basename_heatmap)

    # Save heatmap_probabilities as .txt file
    np.savetxt(full_path_heatmap, probabilities)

def get_mesh_path(base_path, new_subpath):
    # Convert base_path to a Path object
    base_path = Path(base_path)

    # Extract the root part (up to 'Shared drives')
    shared_drives_root = str(base_path).split('Shared drives')[0] + 'Shared drives'

    # Create the new path by appending new_subpath
    amended_path = Path(shared_drives_root) / new_subpath

    return str(amended_path)

def process_file_list(config, mesh_path):
    print('Processing filelist ', mesh_path)
    names = []
    with open(mesh_path) as f:
        for line in f:
            line = (line.strip("/n")).strip("\n")
            if len(line) > 4:
                names.append(line)
    print('Processing ', len(names), ' meshes')
    dm = deepmvlm.DeepMVLM(config)
    for mesh_path in names:
        print('Processing ', mesh_path)
        name_lm_txt = os.path.splitext(mesh_path)[0] + '_landmarks.txt'
        landmarks = dm.predict_one_file(mesh_path)
        dm.write_landmarks_as_text(landmarks, name_lm_txt)


def process_files_in_dir(config, dir_name):
    print('Processing files in  ', dir_name)
    names = Utils3D.get_mesh_files_in_dir(dir_name)
    print('Processing ', len(names), ' meshes')
    dm = deepmvlm.DeepMVLM(config)
    for mesh_path in names:
        print('Processing ', mesh_path)
        name_lm_txt = os.path.splitext(mesh_path)[0] + '_landmarks.txt'
        landmarks = dm.predict_one_file(mesh_path)
        dm.write_landmarks_as_text(landmarks, name_lm_txt)


def main(config):
    name = str(config.name)
    if name.lower().endswith(('.obj', '.wrl', '.vtk', '.vtp', '.ply', '.stl')) and os.path.isfile(name):
        process_one_file(config, name)
    elif name.lower().endswith('.txt') and os.path.isfile(name):
        process_file_list(config, name)
    elif os.path.isdir(name):
        process_files_in_dir(config, name)
    else:
        print('Cannot process (not a mesh file, a filelist (.txt) or a directory)', name)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Deep-MVLM')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-n', '--name', default=None, type=str,
                      help='name of file, filelist (.txt) or directory to be processed')

    global_config = ConfigParser(args)
    main(global_config)
