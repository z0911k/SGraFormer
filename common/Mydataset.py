import torch
import numpy as np
import torch.utils.data as data

from common.cameras import normalize_screen_coordinates


class ChunkedGenerator:

    def __init__(self, batch_size, cameras, poses_3d, poses_2d,
                 chunk_length=1, pad=0, causal_shift=0,
                 shuffle=False, random_seed=1234,
                 augment=False, reverse_aug=False, kps_left=None, kps_right=None, joints_left=None, joints_right=None,
                 endless=False, out_all=False):
        assert poses_3d is None or len(poses_3d) == len(poses_2d), (len(poses_3d), len(poses_2d))
        assert cameras is None or len(cameras) == len(poses_2d)

        pairs = []
        self.saved_index = {}
        start_index = 0

        for key in poses_2d.keys():
            assert poses_3d is None or poses_2d[key].shape[0] == poses_3d[key].shape[0]
            n_chunks = (poses_2d[key].shape[0] + chunk_length - 1) // chunk_length
            offset = (n_chunks * chunk_length - poses_2d[key].shape[0]) // 2
            bounds = np.arange(n_chunks + 1) * chunk_length - offset
            augment_vector = np.full(len(bounds - 1), False, dtype=bool)
            reverse_augment_vector = np.full(len(bounds - 1), False, dtype=bool)
            keys = np.tile(np.array(key).reshape([1, 2]), (len(bounds - 1), 1))
            pairs += list(zip(keys, bounds[:-1], bounds[1:], augment_vector, reverse_augment_vector))
            if reverse_aug:
                pairs += list(zip(keys, bounds[:-1], bounds[1:], augment_vector, ~reverse_augment_vector))
            if augment:
                if reverse_aug:
                    pairs += list(zip(keys, bounds[:-1], bounds[1:], ~augment_vector, ~reverse_augment_vector))
                else:
                    pairs += list(zip(keys, bounds[:-1], bounds[1:], ~augment_vector, reverse_augment_vector))
            end_index = start_index + poses_3d[key].shape[0]
            self.saved_index[key] = [start_index, end_index]
            start_index = start_index + poses_3d[key].shape[0]

        if cameras is not None:
            self.batch_cam = np.empty((batch_size, cameras[key].shape[-1]))

        if poses_3d is not None:
            self.batch_3d = np.empty((batch_size, chunk_length, poses_3d[key].shape[-2], poses_3d[key].shape[-1]))
        self.batch_2d = np.empty(
            (batch_size, chunk_length + 2 * pad, poses_2d[key].shape[-3], poses_2d[key].shape[-2],
             poses_2d[key].shape[-1]))

        self.num_batches = (len(pairs) + batch_size - 1) // batch_size
        self.batch_size = batch_size
        self.random = np.random.RandomState(random_seed)
        self.pairs = pairs
        self.shuffle = shuffle
        self.pad = pad
        self.causal_shift = causal_shift
        self.endless = endless
        self.state = None

        self.cameras = cameras
        if cameras is not None:
            self.cameras = cameras
        self.poses_3d = poses_3d
        self.poses_2d = poses_2d

        self.augment = augment
        self.kps_left = kps_left
        self.kps_right = kps_right
        self.joints_left = joints_left
        self.joints_right = joints_right
        self.out_all = out_all

    def num_frames(self):
        return self.num_batches * self.batch_size

    def random_state(self):
        return self.random

    def set_random_state(self, random):
        self.random = random

    def augment_enabled(self):
        return self.augment

    def next_pairs(self):
        if self.state is None:
            if self.shuffle:
                pairs = self.random.permutation(self.pairs)
            else:
                pairs = self.pairs
            return 0, pairs
        else:
            return self.state

    def get_batch(self, seq_i, start_3d, end_3d, flip, reverse):
        subject, action = seq_i
        seq_name = (subject, action)
        start_2d = start_3d - self.pad - self.causal_shift  # \u5f00\u59cb\u4f4d\u7f6e
        end_2d = end_3d + self.pad - self.causal_shift

        seq_2d = self.poses_2d[seq_name].copy()
        low_2d = max(start_2d, 0)
        high_2d = min(end_2d, seq_2d.shape[0])
        pad_left_2d = low_2d - start_2d
        pad_right_2d = end_2d - high_2d

        if pad_left_2d != 0 or pad_right_2d != 0:
            self.batch_2d = np.pad(seq_2d[low_2d:high_2d], ((pad_left_2d, pad_right_2d), (0, 0), (0, 0), (0, 0)),
                                   'edge')
        else:
            self.batch_2d = seq_2d[low_2d:high_2d]

        if flip:
            self.batch_2d[:, :, :, 0] *= -1
            self.batch_2d[:, :, self.kps_left + self.kps_right] = self.batch_2d[:, :, self.kps_right + self.kps_left]

        if reverse:
            self.batch_2d = self.batch_2d[::-1].copy()

        if self.poses_3d is not None:
            seq_3d = self.poses_3d[seq_name].copy()
            if self.out_all:
                low_3d = low_2d
                high_3d = high_2d
                pad_left_3d = pad_left_2d
                pad_right_3d = pad_right_2d
            else:
                low_3d = max(start_3d, 0)
                high_3d = min(end_3d, seq_3d.shape[0])
                pad_left_3d = low_3d - start_3d
                pad_right_3d = end_3d - high_3d
            if pad_left_3d != 0 or pad_right_3d != 0:
                self.batch_3d = np.pad(seq_3d[low_3d:high_3d],
                                       ((pad_left_3d, pad_right_3d), (0, 0), (0, 0)), 'edge')
            else:
                self.batch_3d = seq_3d[low_3d:high_3d]

            if flip:
                self.batch_3d[:, :, 0] *= -1
                self.batch_3d[:, self.joints_left + self.joints_right] = \
                    self.batch_3d[:, self.joints_right + self.joints_left]
            if reverse:
                self.batch_3d = self.batch_3d[::-1].copy()

        if self.poses_3d is None and self.cameras is None:
            return None, None, self.batch_2d.copy(), action, subject
        elif self.poses_3d is not None and self.cameras is None:
            return np.zeros(9), self.batch_3d.copy(), self.batch_2d.copy(), action, subject, low_2d, high_2d
        elif self.poses_3d is None:
            return self.batch_cam, None, self.batch_2d.copy(), action, subject
        else:
            return self.batch_cam, self.batch_3d.copy(), self.batch_2d.copy(), action, subject


class Fusion(data.Dataset):
    def __init__(self, opt, dataset, root_path, train=True):
        self.hop1 = torch.tensor([[0, 1,	0,	0,	1,	0,	0,	1,	0,	0,	0,	0,	0,	0,	0,	0,	0],
                            [1,	0,	1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0],
                            [0,	1,	0,	1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0],
                            [0,	0,	1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0],
                            [1,	0,	0,	0,	0,	1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0],
                            [0,	0,	0,	0,	1,	0,	1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0],
                            [0,	0,	0,	0,	0,	1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0],
                            [1,	0,	0,	0,	0,	0,	0,	0,	1,	0,	0,	0,	0,	0,	0,	0,	0],
                            [0,	0,	0,	0,	0,	0,	0,	1,	0,	1,	0,	1,	0,	0,	1,	0,	0],
                            [0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	1,	0,	0,	0,	0,	0,	0],
                            [0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	0,	0,	0,	0,	0,	0],
                            [0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	0,	0,	1,	0,	0,	0,	0],
                            [0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	1,	0,	0,	0],
                            [0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	0,	0,	0],
                            [0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	0,	0,	0,	0,	0,	1,	0],
                            [0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	1],
                            [0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	0]])

        self.hop2 = torch.tensor([[0,	0,	1,	0,	0,	1,	0,	0,	1,	0,	0,	0,	0,	0,	0,	0,	0],
                            [0,	0,	0,	1,	1,	0,	0,	1,	0,	0,	0,	0,	0,	0,	0,	0,	0],
                            [1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0],
                            [0,	1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0],
                            [0,	1,	0,	0,	0,	0,	1,	1,	0,	0,	0,	0,	0,	0,	0,	0,	0],
                            [1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0],
                            [0,	0,	0,	0,	1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0],
                            [0,	1,	0,	0,	1,	0,	0,	0,	0,	1,	0,	1,	0,	0,	1,	0,	0],
                            [1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	1,	0,	0,	1,	0],
                            [0,	0,	0,	0,	0,	0,	0,	1,	0,	0,	0,	1,	0,	0,	1,	0,	0],
                            [0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	0,	0,	0,	0,	0,	0,	0],
                            [0,	0,	0,	0,	0,	0,	0,	1,	0,	1,	0,	0,	0,	1,	1,	0,	0],
                            [0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	0,	0,	0,	0,	0,	0,	0],
                            [0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	0,	0,	0,	0],
                            [0,	0,	0,	0,	0,	0,	0,	1,	0,	1,	0,	0,	0,	0,	0,	0,	1],
                            [0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	0,	0,	0,	0,	0,	0,	0],
                            [0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	0]])

        self.hop3 = torch.tensor([[0,	0,	0,	1,	0,	0,	1,	0,	0,	1,	0,	1,	0,	0,	1,	0,	0],
                            [0,	0,	0,	0,	0,	1,	0,	0,	1,	0,	0,	0,	0,	0,	0,	0,	0],
                            [0,	0,	0,	0,	1,	0,	0,	1,	0,	0,	0,	0,	0,	0,	0,	0,	0],
                            [1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0],
                            [0,	0,	1,	0,	0,	0,	0,	0,	1,	0,	0,	0,	0,	0,	0,	0,	0],
                            [0,	1,	0,	0,	0,	0,	0,	1,	0,	0,	0,	0,	0,	0,	0,	0,	0],
                            [1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0],
                            [0,	0,	1,	0,	0,	1,	0,	0,	0,	0,	1,	0,	1,	0,	0,	1,	0],
                            [0,	1,	0,	0,	1,	0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	0,	1],
                            [1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	0,	1,	0],
                            [0,	0,	0,	0,	0,	0,	0,	1,	0,	0,	0,	1,	0,	0,	1,	0,	0],
                            [1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	0,	0,	0,	1,	0],
                            [0,	0,	0,	0,	0,	0,	0,	1,	0,	1,	0,	0,	0,	0,	1,	0,	0],
                            [0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	0,	0,	0,	0,	0,	0,	0],
                            [1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	1,	0,	0,	0,	0],
                            [0,	0,	0,	0,	0,	0,	0,	1,	0,	1,	0,	1,	0,	0,	0,	0,	0],
                            [0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	0,	0,	0,	0,	0,	0,	0]])

        self.hop4 = torch.tensor([[0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	1,	0,	0,	1,	0],
                            [0,	0,	0,	0,	0,	0,	1,	0,	0,	1,	0,	1,	0,	0,	1,	0,	0],
                            [0,	0,	0,	0,	0,	1,	0,	0,	1,	0,	0,	0,	0,	0,	0,	0,	0],
                            [0,	0,	0,	0,	1,	0,	0,	1,	0,	0,	0,	0,	0,	0,	0,	0,	0],
                            [0,	0,	0,	1,	0,	0,	0,	0,	0,	1,	0,	1,	0,	0,	1,	0,	0],
                            [0,	0,	1,	0,	0,	0,	0,	0,	1,	0,	0,	0,	0,	0,	0,	0,	0],
                            [0,	1,	0,	0,	0,	0,	0,	1,	0,	0,	0,	0,	0,	0,	0,	0,	0],
                            [0,	0,	0,	1,	0,	0,	1,	0,	0,	0,	0,	0,	0,	1,	0,	0,	1],
                            [0,	0,	1,	0,	0,	1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0],
                            [0,	1,	0,	0,	1,	0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	0,	1],
                            [1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	0,	1,	0],
                            [0,	1,	0,	0,	1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1],
                            [1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	0,	0,	0,	1,	0],
                            [0,	0,	0,	0,	0,	0,	0,	1,	0,	1,	0,	0,	0,	0,	1,	0,	0],
                            [0,	1,	0,	0,	1,	0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	0,	0],
                            [1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	1,	0,	0,	0,	0],
                            [0,	0,	0,	0,	0,	0,	0,	1,	0,	1,	0,	1,	0,	0,	0,	0,	0]])


        self.data_type = opt.dataset
        self.train = train
        self.keypoints_name = opt.keypoints
        self.root_path = root_path

        self.train_list = opt.subjects_train.split(',')
        self.test_list = opt.subjects_test.split(',')
        self.action_filter = None if opt.actions == '*' else opt.actions.split(',')
        self.downsample = opt.downsample
        self.subset = opt.subset
        self.stride = opt.stride
        self.crop_uv = opt.crop_uv
        self.test_aug = opt.test_augmentation
        self.pad = opt.pad

        if self.train:
            self.keypoints = self.prepare_data(dataset, self.train_list)
            self.cameras_train, self.poses_train, self.poses_train_2d = self.fetch(dataset, self.train_list,
                                                                                   subset=self.subset)
            self.generator = ChunkedGenerator(opt.batch_size // opt.stride, self.cameras_train, self.poses_train,
                                              self.poses_train_2d, self.stride, pad=self.pad,
                                              augment=opt.data_augmentation, reverse_aug=opt.reverse_augmentation,
                                              kps_left=self.kps_left, kps_right=self.kps_right,
                                              joints_left=self.joints_left,
                                              joints_right=self.joints_right, out_all=opt.out_all)
            print('INFO: Training on {} frames'.format(self.generator.num_frames()))
        else:
            self.keypoints = self.prepare_data(dataset, self.test_list)
            self.cameras_test, self.poses_test, self.poses_test_2d = self.fetch(dataset, self.test_list,
                                                                                subset=self.subset)
            self.generator = ChunkedGenerator(opt.batch_size // opt.stride, self.cameras_test, self.poses_test,
                                              self.poses_test_2d,
                                              pad=self.pad, augment=False, kps_left=self.kps_left,
                                              kps_right=self.kps_right, joints_left=self.joints_left,
                                              joints_right=self.joints_right)
            self.key_index = self.generator.saved_index
            print('INFO: Testing on {} frames'.format(self.generator.num_frames()))

    def prepare_data(self, dataset, folder_list):

        for subject in folder_list:
            for action in dataset[subject].keys():
                dataset[subject][action]['positions'][:, 1:] -= dataset[subject][action]['positions'][:, :1]

        keypoints = np.load(self.root_path + 'data_2d_' + self.data_type + '_' + self.keypoints_name + '.npz',
                            allow_pickle=True)
        keypoints_symmetry = keypoints['metadata'].item()['keypoints_symmetry']

        self.kps_left, self.kps_right = list(keypoints_symmetry[0]), list(keypoints_symmetry[1])
        self.joints_left, self.joints_right = list(dataset.skeleton().joints_left()), list(
            dataset.skeleton().joints_right())

        keypoints = keypoints['positions_2d'].item()

        for subject in folder_list:
            for action in dataset[subject].keys():
                mocap_length = dataset[subject][action]['positions'].shape[0]
                for cam_idx in range(len(keypoints[subject][action])):
                    assert keypoints[subject][action][cam_idx].shape[0] >= mocap_length
                    if keypoints[subject][action][cam_idx].shape[0] > mocap_length:
                        keypoints[subject][action][cam_idx] = keypoints[subject][action][cam_idx][:mocap_length]

        for subject in keypoints.keys():
            for action in keypoints[subject]:
                for cam_idx, kps in enumerate(keypoints[subject][action]):
                    cam = dataset.cameras()[subject][cam_idx]
                    if self.crop_uv == 0:
                        kps[..., :2] = normalize_screen_coordinates(kps[..., :2], w=cam['res_w'], h=cam['res_h'])
                    keypoints[subject][action][cam_idx] = kps

        for subject in folder_list:
            for action in dataset[subject].keys():
                positions_2d_pairs = []
                for cam_idx in range(len(keypoints[subject][action])):
                    positions_2d_pairs.append(keypoints[subject][action][cam_idx])

                keypoints[subject][action].append(
                    np.array(positions_2d_pairs).transpose((1, 0, 2,3)))
        return keypoints

    def fetch(self, dataset, subjects, subset=1, ):
        out_poses_3d = {}
        out_poses_2d = {}
        out_camera_params = {}
        for subject in subjects:
            for action in self.keypoints[subject].keys():
                poses_2d = self.keypoints[subject][action][4]
                out_poses_2d[(subject, action)] = poses_2d

                poses_3d = dataset[subject][action]['positions']
                out_poses_3d[(subject, action)] = poses_3d

        if len(out_camera_params) == 0:
            out_camera_params = None

        downsample = 1
        if downsample:
            pass
        return out_camera_params, out_poses_3d, out_poses_2d
    
    def hop_normalize(self, x1, x2, x3, x4):
        x1 = x1 / torch.sum(x1, dim=1)
        x2 = x2 / torch.sum(x1, dim=1)
        x3 = x3 / torch.sum(x1, dim=1)
        x4 = x4 / torch.sum(x1, dim=1)
        return torch.cat((x1.unsqueeze(0), x2.unsqueeze(0), x3.unsqueeze(0), x4.unsqueeze(0)), dim=0)

    def __len__(self):
        return len(self.generator.pairs)

    def __getitem__(self, index):
        seq_name, start_3d, end_3d, flip, reverse = self.generator.pairs[index]

        cam, gt_3D, input_2D, action, subject, low_2d, high_2d = self.generator.get_batch(seq_name, start_3d, end_3d,
                                                                                          False, False)

        if self.train == False and self.test_aug:
            _, _, input_2D_aug, _, _, _, _ = self.generator.get_batch(seq_name, start_3d, end_3d, flip=False,
                                                                      reverse=False)
            input_2D = np.concatenate((np.expand_dims(input_2D, axis=0), np.expand_dims(input_2D_aug, axis=0)), 0)

        bb_box = np.array([0, 0, 1, 1])
        input_2D_update = input_2D

        hops = self.hop_normalize(self.hop1, self.hop2, self.hop3, self.hop4)

        scale = np.float64(1.0)

        return cam, gt_3D, input_2D_update, action, subject, scale, bb_box, low_2d, high_2d, hops

