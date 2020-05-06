import os

import numpy as np
import torch
from PIL import Image
import torch.utils.data as data


def random_clip(video_frames, sampling_rate, frames_per_clip, fixed_offset=False):
    """

    Args:
        video_frames (int): total frame number of a video
        sampling_rate (int): sampling rate for clip, pick one every k frames
        frames_per_clip (int): number of frames of a clip
        fixed_offset (bool): used with sample offset to decide the offset value deterministically.

    Returns:
        list[int]: frame indices (started from zero)
    """
    new_sampling_rate = sampling_rate
    highest_idx = video_frames - new_sampling_rate * frames_per_clip
    if highest_idx <= 0:
        random_offset = 0
    else:
        if fixed_offset:
            random_offset = (video_frames - new_sampling_rate * frames_per_clip) // 2
        else:
            random_offset = int(np.random.randint(0, highest_idx, 1))
    frame_idx = [int(random_offset + i * sampling_rate) % video_frames for i in range(frames_per_clip)]
    return frame_idx


class VideoRecord(object):
    def __init__(self, path, start_frame, end_frame, label, reverse=False):
        self.path = path
        self.video_id = os.path.basename(path)
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.label = label
        self.reverse = reverse

    @property
    def num_frames(self):
        return self.end_frame - self.start_frame + 1


class VideoDataSet(data.Dataset):

    def __init__(self, root_path, list_file, num_groups=64, frames_per_group=4, sample_offset=0, num_clips=1,
                 modality='rgb', dense_sampling=False, fixed_offset=True,
                 image_tmpl='{:05d}.jpg', transform=None, is_train=True, test_mode=False, seperator=' ',
                 filter_video=0, num_classes=None):
        """

        Argments have different meaning when dense_sampling is True:
            - num_groups ==> number of frames
            - frames_per_group ==> sample every K frame
            - sample_offset ==> number of clips used in validation or test mode

        Args:
            root_path (str): the file path to the root of video folder
            list_file (str): the file list, each line with folder_path, start_frame, end_frame, label_id
            num_groups (int): number of frames per data sample
            frames_per_group (int): number of frames within one group
            sample_offset (int): used in validation/test, the offset when sampling frames from a group
            modality (str): rgb or flow
            dense_sampling (bool): dense sampling in I3D
            fixed_offset (bool): used for generating the same videos used in TSM
            image_tmpl (str): template of image ids
            transform: the transformer for preprocessing
            is_train (bool): shuffle the video but keep the causality
            test_mode (bool): testing mode, no label
        """
        if modality not in ['flow', 'rgb']:
            raise ValueError("modality should be 'flow' or 'rgb'.")

        self.root_path = root_path
        self.list_file = list_file
        self.num_groups = num_groups
        self.num_frames = num_groups
        self.frames_per_group = frames_per_group
        self.sample_freq = frames_per_group
        self.num_clips = num_clips
        self.sample_offset = sample_offset
        self.fixed_offset = fixed_offset
        self.dense_sampling = dense_sampling
        self.modality = modality.lower()
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.is_train = is_train
        self.test_mode = test_mode
        self.seperator = seperator
        self.filter_video = filter_video

        if self.modality == 'flow':
            self.num_consecutive_frames = 5
        else:
            self.num_consecutive_frames = 1

        self.video_list = self._parse_list()
        self.num_classes = num_classes

    def _image_path(self, directory, idx):
        return os.path.join(self.root_path, directory, self.image_tmpl.format(idx))

    def _load_image(self, directory, idx):

        def _safe_load_image(img_path):
            img_tmp = Image.open(img_path)
            img = img_tmp.copy()
            img_tmp.close()
            return img

        num_try = 0
        image_path_file = os.path.join(self.root_path, directory, self.image_tmpl.format(idx))
        img = None
        while num_try < 10:
            try:
                if self.modality == 'rgb':
                    img = [_safe_load_image(image_path_file)]
                else:
                    ext = image_path_file.split(".")[-1]
                    flow_x_name = image_path_file.replace(".{}".format(ext), "_x.{}".format(ext))
                    flow_y_name = image_path_file.replace(".{}".format(ext), "_y.{}".format(ext))
                    img = [_safe_load_image(flow_x_name), _safe_load_image(flow_y_name)]
                break
            except Exception as e:
                print('[Will try load again] error loading image: {}, error: {}'.format(image_path_file, str(e)))
                num_try += 1

        if img is None:
            raise ValueError('[Fail 10 times] error loading image: {}'.format(image_path_file))

        return img

    def _parse_list(self):
        # usualy it is [video_id, num_frames, class_idx]
        # or [video_id, start_frame, end_frame, list of class_idx]
        tmp = []
        original_video_numbers = 0
        for x in open(self.list_file):
            elements = x.strip().split(self.seperator)
            start_frame = int(elements[1])
            end_frame = int(elements[2])
            total_frame = end_frame - start_frame + 1
            original_video_numbers += 1
            if self.test_mode:
                tmp.append(elements)
            else:
                if total_frame >= self.filter_video:
                    tmp.append(elements)

        num = len(tmp)
        print("The number of videos is {} (with more than {} frames) "
              "(original: {})".format(num, self.filter_video, original_video_numbers), flush=True)
        assert (num > 0)
        file_list = []
        for item in tmp:
            if self.test_mode:
                file_list.append([item[0], int(item[1]), int(item[2]), -1])
            else:
                labels = []
                for i in range(3, len(item)):
                    labels.append(float(item[i]))
                labels = labels[0] if len(labels) == 1 else labels
                file_list.append([item[0], int(item[1]), int(item[2]), labels])

        video_list = [VideoRecord(item[0], item[1], item[2], item[3]) for item in file_list]
        # flow model has one frame less
        if self.modality == 'flow':
            for i in range(len(video_list)):
                video_list[i].end_frame -= 1

        return video_list

    def _sample_indices(self, record):
        """
        Used for training.

        Args:
            - record (VideoRecord):

        Returns:
            list: frame index, index starts from 1.
        """
        max_frame_idx = max(1, record.num_frames - self.num_consecutive_frames + 1)
        if self.dense_sampling:
            frame_idx = np.asarray(random_clip(max_frame_idx, self.sample_freq, self.num_frames))
        else:
            total_frames = self.num_groups * self.frames_per_group
            ave_frames_per_group = max_frame_idx // self.num_groups
            if ave_frames_per_group >= self.frames_per_group:
                # randomly sample f images per segement
                frame_idx = np.arange(0, self.num_groups) * ave_frames_per_group
                frame_idx = np.repeat(frame_idx, repeats=self.frames_per_group)
                offsets = np.random.choice(ave_frames_per_group, self.frames_per_group, replace=False)
                offsets = np.tile(offsets, self.num_groups)
                frame_idx = frame_idx + offsets
            elif max_frame_idx < total_frames:
                # need to sample the same images
                frame_idx = np.random.choice(max_frame_idx, total_frames)
            else:
                # sample cross all images
                frame_idx = np.random.choice(max_frame_idx, total_frames, replace=False)
            frame_idx = np.sort(frame_idx)
        frame_idx = frame_idx + 1
        return frame_idx

    def _get_val_indices(self, record):
        max_frame_idx = max(1, record.num_frames - self.num_consecutive_frames + 1)
        if self.dense_sampling:
            if self.fixed_offset:
                sample_pos = max(1, 1 + max_frame_idx - self.sample_freq * self.num_frames)
                t_stride = self.sample_freq
                start_list = np.linspace(0, sample_pos - 1, num=self.num_clips, dtype=int)
                frame_idx = []
                for start_idx in start_list.tolist():
                    frame_idx += [(idx * t_stride + start_idx) % max_frame_idx for idx in range(self.num_frames)]
            else:
                frame_idx = []
                for i in range(self.num_clips):
                    frame_idx.extend(random_clip(max_frame_idx, self.sample_freq, self.num_frames))
            frame_idx = np.asarray(frame_idx) + 1
        else:  # uniform sampling
            if self.fixed_offset:
                frame_idices = []
                sample_offsets = list(range(-self.num_clips // 2 + 1, self.num_clips // 2 + 1))
                for sample_offset in sample_offsets:
                    if max_frame_idx > self.num_groups:
                        tick = max_frame_idx / float(self.num_groups)
                        curr_sample_offset = sample_offset
                        if curr_sample_offset >= tick / 2.0:
                            curr_sample_offset = tick / 2.0 - 1e-4
                        elif curr_sample_offset < -tick / 2.0:
                            curr_sample_offset = -tick / 2.0
                        frame_idx = np.array([int(tick / 2.0 + curr_sample_offset + tick * x) for x in range(self.num_groups)])
                    else:
                        np.random.seed(sample_offset - (-self.num_clips // 2 + 1))
                        frame_idx = np.random.choice(max_frame_idx, self.num_groups)
                    frame_idx = np.sort(frame_idx)
                    frame_idices.extend(frame_idx.tolist())
            else:
                frame_idices = []
                for i in range(self.num_clips):
                    total_frames = self.num_groups * self.frames_per_group
                    ave_frames_per_group = max_frame_idx // self.num_groups
                    if ave_frames_per_group >= self.frames_per_group:
                        # randomly sample f images per segment
                        frame_idx = np.arange(0, self.num_groups) * ave_frames_per_group
                        frame_idx = np.repeat(frame_idx, repeats=self.frames_per_group)
                        offsets = np.random.choice(ave_frames_per_group, self.frames_per_group, replace=False)
                        offsets = np.tile(offsets, self.num_groups)
                        frame_idx = frame_idx + offsets
                    elif max_frame_idx < total_frames:
                        # need to sample the same images
                        np.random.seed(i)
                        frame_idx = np.random.choice(max_frame_idx, total_frames)
                    else:
                        # sample cross all images
                        np.random.seed(i)
                        frame_idx = np.random.choice(max_frame_idx, total_frames, replace=False)
                    frame_idx = np.sort(frame_idx)
                    frame_idices.extend(frame_idx.tolist())
            frame_idx = np.asarray(frame_idices) + 1
        return frame_idx

    def __getitem__(self, index):
        """
        Returns:
            torch.FloatTensor: (3xgxf)xHxW dimension, g is number of groups and f is the frames per group.
            torch.FloatTensor: the label
        """
        record = self.video_list[index]
        # check this is a legit video folder
        if self.is_train:
            indices = self._sample_indices(record)
        else:
            indices = self._get_val_indices(record)

        images = []
        for seg_ind in indices:
            for i in range(self.num_consecutive_frames):
                new_seg_ind = min(seg_ind + record.start_frame - 1 + i, record.num_frames)
                seg_imgs = self._load_image(record.path, new_seg_ind)
                images.extend(seg_imgs)

        images = self.transform(images)
        if self.test_mode:
            # in test mode, return the video id as label
            label = int(record.video_id)
        else:
            label = int(record.label)

        # re-order data to targeted format.
        return images, label

    def __len__(self):
        return len(self.video_list)
