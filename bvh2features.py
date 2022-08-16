# This code was written by Simon Alexanderson
# and is released here: https://github.com/simonalexanderson/PyMO

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from argparse import ArgumentParser

import glob
import os
import sys

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from pymo.parsers import BVHParser
from pymo.data import Joint, MocapData
from pymo.preprocessing import *
from pymo.viz_tools import *
from pymo.writers import *

import joblib as jl
import glob


def extract_joint_angles(bvh_dir, files, dest_dir, pipeline_dir, fps):
    p = BVHParser()

    data_all = list()
    for f in files:
        ff = os.path.join(bvh_dir, f)
        print(ff)
        data_all.append(p.parse(ff))

    data_pipe = Pipeline([
        # ('dwnsampl', DownSampler(tgt_fps=fps,  keep_all=False)),
        # ('mir', Mirror(axis='X', append=True)),
        ('exp', MocapParameterizer('expmap')),
        # ('root', RootTransformer('hip_centric')),
        ('np', Numpyfier())
    ])

    out_data = data_pipe.fit_transform(data_all)

    # the datapipe will append the mirrored files to the end
    assert len(out_data) == len(files)

    jl.dump(data_pipe, os.path.join(pipeline_dir + 'data_pipe.sav'))

    fi = 0
    for f in files:
        ff = os.path.join(dest_dir, f)
        print(ff)
        np.savez(ff[:-4] + ".npz", clips=out_data[fi])
        # np.savez(ff[:-4] + "_mirrored.npz", clips=out_data[len(files)+fi])
        fi = fi + 1


class RootNormalizer(BaseEstimator, TransformerMixin):
    """
    Make subjects in TalkingWithHands16.2M face the same direction
    This class is not for general uses. Only compatible to GENEA 2022 challenge dataset
    Added by Youngwoo Yoon, April 2022
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        print("RootNormalizer")
        Q = []

        for track in X:
            new_track = track.clone()

            xp_col = '%s_Xposition'%track.root_name
            yp_col = '%s_Yposition'%track.root_name
            zp_col = '%s_Zposition'%track.root_name

            xr_col = '%s_Xrotation'%track.root_name
            yr_col = '%s_Yrotation'%track.root_name
            zr_col = '%s_Zrotation'%track.root_name

            new_df = track.values.copy()

            all_zeros = np.zeros(track.values[xp_col].values.shape)
            mean_xp = np.mean(track.values[xp_col].values)
            mean_yp = np.mean(track.values[yp_col].values)
            mean_zp = np.mean(track.values[zp_col].values)

            if track.values[xp_col].values[0] < 0:
                new_yr = np.full(track.values[xp_col].values.shape, -90)
            else:
                new_yr = np.full(track.values[xp_col].values.shape, 90)

            new_df[xp_col] = pd.Series(data=track.values[xp_col]-mean_xp, index=new_df.index)
            new_df[yp_col] = pd.Series(data=track.values[yp_col]-mean_yp, index=new_df.index)
            new_df[zp_col] = pd.Series(data=track.values[zp_col]-mean_zp, index=new_df.index)

            new_df[xr_col] = pd.Series(data=all_zeros, index=new_df.index)
            new_df[yr_col] = pd.Series(data=new_yr, index=new_df.index)
            new_df[zr_col] = pd.Series(data=all_zeros, index=new_df.index)

            new_track.values = new_df

            Q.append(new_track)

        return Q

    def inverse_transform(self, X, copy=None):
        # NOT IMPLEMENTED
        return X


target_joints = ['b_spine0', 'b_spine1', 'b_spine2', 'b_spine3', 'b_l_shoulder', 'b_l_arm', 'b_l_arm_twist', 'b_l_forearm', 'b_l_wrist_twist', 'b_l_wrist', 'b_r_shoulder', 'b_r_arm', 'b_r_arm_twist', 'b_r_forearm', 'b_r_wrist_twist', 'b_r_wrist', 'b_neck0', 'b_head']


def process_bvh(gesture_filename, dump_pipeline=False):
    p = BVHParser()

    data_all = list()
    data_all.append(p.parse(gesture_filename))

    data_pipe = Pipeline([
        ('dwnsampl', DownSampler(tgt_fps=30, keep_all=False)),
        ('root', RootNormalizer()),
        ('jtsel', JointSelector(target_joints, include_root=False)),
        # ('mir', Mirror(axis='X', append=True)),
        ('np', Numpyfier())
    ])

    out_data = data_pipe.fit_transform(data_all)
    if dump_pipeline:
        jl.dump(data_pipe, os.path.join('../resource', 'data_pipe.sav'))

    # euler -> rotation matrix
    out_data = out_data.reshape((out_data.shape[0], out_data.shape[1], -1, 6))  # 3 pos (XYZ), 3 rot (ZXY)
    out_matrix = np.zeros((out_data.shape[0], out_data.shape[1], out_data.shape[2], 12))  # 3 pos, 1 rot matrix (9 elements)
    for i in range(out_data.shape[0]):  # mirror
        for j in range(out_data.shape[1]):  # frames
            for k in range(out_data.shape[2]):  # joints
                out_matrix[i, j, k, :3] = out_data[i, j, k, :3]  # positions
                r = R.from_euler('ZXY', out_data[i, j, k, 3:], degrees=True)
                out_matrix[i, j, k, 3:] = r.as_matrix().flatten()  # rotations
    out_matrix = out_matrix.reshape((out_data.shape[0], out_data.shape[1], -1))

    return out_matrix[0]


if __name__ == '__main__':
    # '''
    # python bvh2features.py --bvh_dir <..your path/GENEA/genea_challenge_2022/dataset/TEST/bvh/> --dest_dir <..your path/GENEA/genea_challenge_2022/dataset/TEST/rep/> --pipeline_dir <..your path/GENEA/genea_challenge_2022/dataset/TEST/>
    # '''
    # # Setup parameter parser
    # parser = ArgumentParser(add_help=False)
    # parser.add_argument('--bvh_dir', '-orig', required=True,
    #                     help="Path where original motion files (in BVH format) are stored")
    # parser.add_argument('--dest_dir', '-dest', required=True,
    #                     help="Path where extracted motion features will be stored")
    # parser.add_argument('--pipeline_dir', '-pipe', default="./utils/",
    #                     help="Path where the motion data processing pipeline will be stored")
    #
    # params = parser.parse_args()
    #
    # files = []
    # # Go over all BVH files
    # print("Going to pre-process the following motion files:")
    # files = sorted([f for f in glob.iglob(params.bvh_dir + '/*.bvh')])
    #
    # extract_joint_angles(params.bvh_dir, files, params.dest_dir, params.pipeline_dir, fps=30)
    #
    x = "<..your path/GENEA/genea_challenge_2022/baselines/Aud2Repr2Pose/dataset/raw/bvh/trn_2022_v0_011.bvh>"
    x2 = "<..your path/GENEA/genea_challenge_2022/dataset/TEST/bvh/val_2022_v1_000.bvh>"
    y = process_bvh(x2)
    print(y.shape)      # (1889, 276)

    # import numpy as np
    # x = np.load("<..your path/GENEA/genea_challenge_2022/dataset/TEST/bvh/val_2022_v1_000.npz>")
    #
    # print(x)
