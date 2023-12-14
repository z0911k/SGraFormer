import numpy as np
import copy

from common.cameras import h36m_cameras_intrinsic_params, h36m_cameras_extrinsic_params, \
    normalize_screen_coordinates


class Skeleton:

    def __init__(self, parents, joints_left, joints_right):
        assert len(joints_left) == len(joints_right)

        self._parents = np.array(parents)
        self._joints_left = joints_left
        self._joints_right = joints_right
        self._compute_metadata()

    def num_joints(self):
        return len(self._parents)

    def parents(self):
        return self._parents

    def has_children(self):
        return self._has_children

    def children(self):
        return self._children

    def remove_joints(self, joints_to_remove):

        valid_joints = []
        for joint in range(len(self._parents)):
            if joint not in joints_to_remove:
                valid_joints.append(joint)

        for i in range(len(self._parents)):
            while self._parents[i] in joints_to_remove:
                self._parents[i] = self._parents[self._parents[i]]

        index_offsets = np.zeros(len(self._parents), dtype=int)
        new_parents = []
        for i, parent in enumerate(self._parents):
            if i not in joints_to_remove:
                new_parents.append(parent - index_offsets[parent])
            else:
                index_offsets[i:] += 1
        self._parents = np.array(new_parents)

        if self._joints_left is not None:
            new_joints_left = []
            for joint in self._joints_left:
                if joint in valid_joints:
                    new_joints_left.append(joint - index_offsets[joint])
            self._joints_left = new_joints_left
        if self._joints_right is not None:
            new_joints_right = []
            for joint in self._joints_right:
                if joint in valid_joints:
                    new_joints_right.append(joint - index_offsets[joint])
            self._joints_right = new_joints_right

        self._compute_metadata()

        return valid_joints

    def joints_left(self):
        return self._joints_left

    def joints_right(self):
        return self._joints_right

    def _compute_metadata(self):
        self._has_children = np.zeros(len(self._parents)).astype(bool)
        for i, parent in enumerate(self._parents):
            if parent != -1:
                self._has_children[parent] = True

        self._children = []
        for i, parent in enumerate(self._parents):
            self._children.append([])
        for i, parent in enumerate(self._parents):
            if parent != -1:
                self._children[parent].append(i)


h36m_skeleton = Skeleton(parents=[-1, 0, 1, 2, 3, 4, 0, 6, 7, 8, 9, 0, 11, 12, 13, 14, 12,
                                  16, 17, 18, 19, 20, 19, 22, 12, 24, 25, 26, 27, 28, 27, 30],  # 树的双亲表示法
                         joints_left=[6, 7, 8, 9, 10, 16, 17, 18, 19, 20, 21, 22, 23],
                         joints_right=[1, 2, 3, 4, 5, 24, 25, 26, 27, 28, 29, 30, 31])


class MocapDataset:
    def __init__(self, fps, skeleton):
        self._skeleton = skeleton
        self._fps = fps
        self._data = None
        self._cameras = None

    def remove_joints(self, joints_to_remove):
        kept_joints = self._skeleton.remove_joints(joints_to_remove)
        for subject in self._data.keys():
            for action in self._data[subject].keys():
                s = self._data[subject][action]
                s['positions'] = s['positions'][:, kept_joints]

    def __getitem__(self, key):
        return self._data[key]

    def subjects(self):
        return self._data.keys()

    def fps(self):
        return self._fps

    def skeleton(self):
        return self._skeleton

    def cameras(self):
        return self._cameras

    def supports_semi_supervised(self):
        return False


class Human36mDataset(MocapDataset):
    def __init__(self, path, opt, remove_static_joints=True):
        super().__init__(fps=50, skeleton=h36m_skeleton)
        self.train_list = ['S1', 'S5', 'S6', 'S7', 'S8']
        self.test_list = ['S9', 'S11']

        self._cameras = copy.deepcopy(h36m_cameras_extrinsic_params)
        for cameras in self._cameras.values():
            for i, cam in enumerate(cameras):
                cam.update(h36m_cameras_intrinsic_params[i])
                for k, v in cam.items():
                    if k not in ['id', 'res_w', 'res_h']:
                        cam[k] = np.array(v, dtype='float32')

                if opt.crop_uv == 0:
                    cam['center'] = normalize_screen_coordinates(cam['center'], w=cam['res_w'], h=cam['res_h']).astype(
                        'float32')
                    cam['focal_length'] = cam['focal_length'] / cam['res_w'] * 2

                if 'translation' in cam:
                    cam['translation'] = cam['translation'] / 1000

                cam['intrinsic'] = np.concatenate((cam['focal_length'],
                                                   cam['center'],
                                                   cam['radial_distortion'],
                                                   cam['tangential_distortion']))

        data = np.load(path, allow_pickle=True)['positions_3d'].item()

        self._data = {}
        for subject, actions in data.items():
            self._data[subject] = {}
            for action_name, positions in actions.items():
                self._data[subject][action_name] = {
                    'positions': positions,
                    'cameras': self._cameras[subject],
                }

        if remove_static_joints:
            self.remove_joints([4, 5, 9, 10, 11, 16, 20, 21, 22, 23, 24, 28, 29, 30, 31])

            self._skeleton._parents[11] = 8
            self._skeleton._parents[14] = 8

    def supports_semi_supervised(self):
        return True