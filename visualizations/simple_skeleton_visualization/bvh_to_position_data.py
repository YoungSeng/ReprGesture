import glob
from sklearn.pipeline import Pipeline

from pymo.parsers import BVHParser
from pymo.preprocessing import *


def get_joint_tree(path):
    p = BVHParser()
    X = p.parse(path)

    joint_name_to_idx = {}
    for i, joint in enumerate(X.traverse()):
        joint_name_to_idx[joint] = i

    # traverse tree
    joint_links = []
    stack = [X.root_name]
    while stack:
        joint = stack.pop()
        parent = X.skeleton[joint]['parent']
        # tab = len(stack)
        # print('%s- %s (%s)'%('| '*tab, joint, parent))
        if parent:
            joint_links.append((joint_name_to_idx[parent], joint_name_to_idx[joint]))
        for c in X.skeleton[joint]['children']:
            stack.append(c)

    print(joint_name_to_idx)
    print(joint_links)


def process_bvh(gesture_filename):
    p = BVHParser()

    data_all = list()
    data_all.append(p.parse(gesture_filename))

    data_pipe = Pipeline([
        ('dwnsampl', DownSampler(tgt_fps=30, keep_all=False)),
        ('root', RootNormalizer()),
        ('param', MocapParameterizer('position')),
        ('np', Numpyfier())
    ])

    out_data = data_pipe.fit_transform(data_all)
    out_data = out_data[0]

    return out_data


def bvh_to_npy(bvh_path):
    print(bvh_path)
    pos_data = process_bvh(bvh_path)
    npy_path = bvh_path.replace('.bvh', '.npy')
    np.save(npy_path, pos_data)


if __name__ == '__main__':
    # print joint tree information
    # get_joint_tree('data_path/sample.bvh')
    
    root = '<..your path/GENEA/genea_challenge_2022/baselines/Tri/output/infer_sample/>'
    # parse bvh
    use_parallel_processing = False
    files = sorted([f for f in glob.iglob(root + '*.bvh')])
    if use_parallel_processing:
        from joblib import Parallel, delayed
        Parallel(n_jobs=8)(delayed(bvh_to_npy)(bvh_path) for bvh_path in files)
    else:
        for bvh_path in files:
            bvh_to_npy(bvh_path)
