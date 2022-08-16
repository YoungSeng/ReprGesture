import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits import mplot3d

joint_name_to_idx = {'body_world': 0, 'b_root': 1, 'b_l_upleg': 2, 'b_l_leg': 3, 'b_l_foot_twist': 4, 'b_l_foot': 5, 'b_l_foot_Nub': 6, 'b_r_upleg': 7, 'b_r_leg': 8, 'b_r_foot_twist': 9, 'b_r_foot': 10, 'b_r_foot_Nub': 11, 'b_spine0': 12, 'b_spine1': 13, 'b_spine2': 14, 'p_navel': 15, 'p_navel_Nub': 16, 'b_spine3': 17, 'b_l_shoulder': 18, 'p_l_scap': 19, 'b_l_arm': 20, 'b_l_arm_twist': 21, 'b_l_forearm': 22, 'b_l_wrist_twist': 23, 'b_l_wrist': 24, 'b_l_pinky1': 25, 'b_l_pinky2': 26, 'b_l_pinky3': 27, 'b_l_pinky3_Nub': 28, 'b_l_ring1': 29, 'b_l_ring2': 30, 'b_l_ring3': 31, 'b_l_ring3_Nub': 32, 'b_l_middle1': 33, 'b_l_middle2': 34, 'b_l_middle3': 35, 'b_l_middle3_Nub': 36, 'b_l_index1': 37, 'b_l_index2': 38, 'b_l_index3': 39, 'b_l_index3_Nub': 40, 'b_l_thumb0': 41, 'b_l_thumb1': 42, 'b_l_thumb2': 43, 'b_l_thumb3': 44, 'b_l_thumb3_Nub': 45, 'p_l_delt': 46, 'p_l_delt_Nub': 47, 'b_r_shoulder': 48, 'p_r_scap': 49, 'b_r_arm': 50, 'b_r_arm_twist': 51, 'b_r_forearm': 52, 'b_r_wrist_twist': 53, 'b_r_wrist': 54, 'b_r_thumb0': 55, 'b_r_thumb1': 56, 'b_r_thumb2': 57, 'b_r_thumb3': 58, 'b_r_thumb3_Nub': 59, 'b_r_pinky1': 60, 'b_r_pinky2': 61, 'b_r_pinky3': 62, 'b_r_pinky3_Nub': 63, 'b_r_middle1': 64, 'b_r_middle2': 65, 'b_r_middle3': 66, 'b_r_middle3_Nub': 67, 'b_r_ring1': 68, 'b_r_ring2': 69, 'b_r_ring3': 70, 'b_r_ring3_Nub': 71, 'b_r_index1': 72, 'b_r_index2': 73, 'b_r_index3': 74, 'b_r_index3_Nub': 75, 'b_neck0': 76, 'b_head': 77, 'b_jaw': 78, 'b_tongue0': 79, 'b_tongue1': 80, 'b_l_tongue1_1': 81, 'b_l_tongue1_1_Nub': 82, 'b_r_tongue1_1': 83, 'b_r_tongue1_1_Nub': 84, 'b_tongue2': 85, 'b_r_tongue2_1': 86, 'b_r_tongue2_1_Nub': 87, 'b_l_tongue2_1': 88, 'b_l_tongue2_1_Nub': 89, 'b_tongue3': 90, 'b_r_tongue3_1': 91, 'b_r_tongue3_1_Nub': 92, 'b_l_tongue3_1': 93, 'b_l_tongue3_1_Nub': 94, 'b_tongue4': 95, 'b_r_tongue4_1': 96, 'b_r_tongue4_1_Nub': 97, 'b_l_tongue4_1': 98, 'b_l_tongue4_1_Nub': 99, 'b_teeth': 100, 'b_teeth_Nub': 101, 'b_jaw_null': 102, 'b_jaw_null_Nub': 103, 'b_r_eye': 104, 'b_r_eye_Nub': 105, 'b_l_eye': 106, 'b_l_eye_Nub': 107, 'b_head_null': 108, 'b_head_null_Nub': 109}
joint_idx_to_name = {v: k for k, v in joint_name_to_idx.items()}
joint_links = [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (1, 7), (7, 8), (8, 9), (9, 10), (10, 11), (1, 12), (12, 13), (13, 14), (14, 15), (15, 16), (14, 17), (17, 18), (18, 19), (19, 20), (20, 21), (21, 22), (22, 23), (23, 24), (24, 25), (25, 26), (26, 27), (27, 28), (24, 29), (29, 30), (30, 31), (31, 32), (24, 33), (33, 34), (34, 35), (35, 36), (24, 37), (37, 38), (38, 39), (39, 40), (24, 41), (41, 42), (42, 43), (43, 44), (44, 45), (18, 46), (46, 47), (17, 48), (48, 49), (49, 50), (50, 51), (51, 52), (52, 53), (53, 54), (54, 55), (55, 56), (56, 57), (57, 58), (58, 59), (54, 60), (60, 61), (61, 62), (62, 63), (54, 64), (64, 65), (65, 66), (66, 67), (54, 68), (68, 69), (69, 70), (70, 71), (54, 72), (72, 73), (73, 74), (74, 75), (17, 76), (76, 77), (77, 78), (78, 79), (79, 80), (80, 81), (81, 82), (80, 83), (83, 84), (80, 85), (85, 86), (86, 87), (85, 88), (88, 89), (85, 90), (90, 91), (91, 92), (90, 93), (93, 94), (90, 95), (95, 96), (96, 97), (95, 98), (98, 99), (78, 100), (100, 101), (78, 102), (102, 103), (77, 104), (104, 105), (77, 106), (106, 107), (77, 108), (108, 109)]


def visualize(x, save_path=None):
    fig = plt.figure(figsize=(4, 3))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.view_init(elev=20, azim=-60)
    plt.tight_layout()

    def animate(i):
        pose = x[i]
        ax.clear()

        for j, pair in enumerate(joint_links):
            ax.plot([pose[pair[0], 0], pose[pair[1], 0]],
                    [pose[pair[0], 2], pose[pair[1], 2]],
                    [pose[pair[0], 1], pose[pair[1], 1]],
                    zdir='z', linewidth=3)
        lim = 150
        ax.set_xlim3d(-lim, lim)
        ax.set_ylim3d(lim, -lim)
        ax.set_zlim3d(0, lim*2)
        ax.set_xlabel('dim 0')
        ax.set_ylabel('dim 2')
        ax.set_zlabel('dim 1')
        ax.margins(x=0)

    num_frames = len(x)

    if save_path:
        ani = animation.FuncAnimation(fig, animate, interval=1, frames=num_frames, repeat=False)
        ani.save(save_path, fps=30, dpi=150)
        del ani
        plt.close(fig)
    else:
        ani = animation.FuncAnimation(fig, animate, interval=5, frames=num_frames, repeat=False)
        plt.show()


def visualize_main():
    root = '<..your path/GENEA/genea_challenge_2022/baselines/Tri/output/infer_sample/>'
    files = sorted([f for f in glob.iglob(root + '*.npy')])
    for i, npy_path in enumerate(files):
        print(npy_path)

        mp4_path = npy_path.replace('.npy', '.mp4')

        x = np.load(npy_path)
        x = x.reshape((x.shape[0], -1, 3))

        # visualize(x)  # show animation
        visualize(x, mp4_path)  # save to mp4


if __name__ == '__main__':
    visualize_main()
