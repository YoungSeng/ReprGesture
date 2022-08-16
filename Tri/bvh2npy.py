# This code was written by Simon Alexanderson
# and is released here: https://github.com/simonalexanderson/PyMO
import pdb

import matplotlib.pyplot as plt
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

    print(1)
    target_joints = ['b_spine0', 'b_spine1', 'b_spine2', 'b_spine3', 'b_l_shoulder', 'b_l_arm', 'b_l_arm_twist',
                     'b_l_forearm', 'b_l_wrist_twist', 'b_l_wrist', 'b_r_shoulder', 'b_r_arm', 'b_r_arm_twist',
                     'b_r_forearm', 'b_r_wrist_twist', 'b_r_wrist', 'b_neck0', 'b_head']
    data_pipe = Pipeline([
        ('dwnsampl', DownSampler(tgt_fps=30, keep_all=False)),
        ('root', RootNormalizer()),
        ('jtsel', JointSelector(target_joints, include_root=False)),
        # ('mir', Mirror(axis='X', append=True)),
        ('np', Numpyfier())
    ])

    print(2)
    out_data = data_pipe.fit_transform(data_all)

    print(3)
    # the datapipe will append the mirrored files to the end
    assert len(out_data) == len(files)

    jl.dump(data_pipe, os.path.join(pipeline_dir + 'data_pipe.sav'))

    fi = 0
    for f in files:
        ff = os.path.join(dest_dir, os.path.basename(f))
        print(ff)
        # np.savez(ff[:-4] + ".npz", clips=out_data[fi])
        np.save(ff[:-4] + ".npy", out_data[fi])
        # np.savez(ff[:-4] + "_mirrored.npz", clips=out_data[len(files)+fi])
        fi = fi + 1


if __name__ == '__main__':
    '''
python bvh2npy.py --bvh_dir "<..your path/GENEA/genea_challenge_2022/baselines/Tri/output/infer_sample/output_2_new_wavlm/bvh/>" --dest_dir "<..your path/GENEA/genea_challenge_2022/baselines/Tri/output/infer_sample/output_2_new_wavlm/npy/>" --pipeline_dir <..your path/GENEA/genea_challenge_2022/baselines/Tri/utils/>
    '''
    # Setup parameter parser
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--bvh_dir', '-orig', required=True,
                        help="Path where original motion files (in BVH format) are stored")
    parser.add_argument('--dest_dir', '-dest', required=True,
                        help="Path where extracted motion features will be stored")
    parser.add_argument('--pipeline_dir', '-pipe', default="./utils/",
                        help="Path where the motion data processing pipeline will be stored")

    params = parser.parse_args()

    files = []
    # Go over all BVH files
    print("Going to pre-process the following motion files:")
    files = sorted([f for f in glob.iglob(params.bvh_dir + '/*.bvh')])

    # print(files)
    # print(params.dest_dir)
    extract_joint_angles(params.bvh_dir, files, params.dest_dir, params.pipeline_dir, fps=30)



    '''
    GT "<..your path/GENEA/genea_challenge_2022/dataset/v1_18_1/val/npy_/>"
    ReprGesture "<..your path/GENEA/genea_challenge_2022/baselines/Tri/output/infer_sample/output_2_new_wavlm/npy_/>"
    w/o wavlm "<..your path/GENEA/genea_challenge_2022/baselines/Tri/output/infer_sample/output_2_wo_wavlm_wo_emo_new/npy/>"
    w/o Gan loss "<..your path/GENEA/genea_challenge_2022/baselines/Tri/output/infer_sample/output_2_wo_emo_new_warmup/npy/>"
    w/o domain loss "<..your path/GENEA/genea_challenge_2022/baselines/Tri/output/infer_sample/output_2_MISA_48_nodif_sim/npy/>"
    w/o Repr "<..your path/GENEA/genea_challenge_2022/baselines/Tri/output/infer_sample/output_2_MISA_48_nodif/npy/>"
    '''

    '''
    FGD
    w/o wavlm "<..your path/GENEA/genea_challenge_2022/baselines/Tri/output/infer_sample/output_2_wo_wavlm_wo_emo_new/npy_FGD/>"
    w/o Gan loss "<..your path/GENEA/genea_challenge_2022/baselines/Tri/output/infer_sample/output_2_wo_emo_new_warmup/npy_FGD/>"
    w/o domain loss "<..your path/GENEA/genea_challenge_2022/baselines/Tri/output/infer_sample/output_2_MISA_48_nodif_sim/npy_FGD/>"
    w/o Repr "<..your path/GENEA/genea_challenge_2022/baselines/Tri/output/infer_sample/output_2_MISA_48_nodif/npy_FGD/>"

    '''
