import math
import glob
import json
import os
import os.path

import cv2
import numpy as np
import torch
import torch.utils.data as data_utl


def video_to_tensor(pic):
    """Convert a ``numpy.ndarray`` to tensor.
    Converts a numpy.ndarray (T x H x W x C)
    to a torch.FloatTensor of shape (C x T x H x W)
    
    Args:
         pic (numpy.ndarray): Video to be converted to tensor.
    Returns:
         Tensor: Converted video.
    """
    return torch.from_numpy(pic.transpose([3, 0, 1, 2]))



def load_rgb_frames_from_video(vid_root, vid, start, num, resize=(256, 256)):
    video_path = os.path.join(vid_root, vid + '.mp4')
    pose_path = os.path.join(vid_root.replace('body_crop_videos','body_crop_poses', 1), vid)
    #print (video_path, pose_path)
    pose_files = sorted(glob.glob(pose_path+'/*'))
    vidcap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
    vidcap.set(cv2.CAP_PROP_POS_FRAMES, start)
    #print (total_frames, len(pose_files))
    pose_files = pose_files[start:]
    poses = []
    #print (num, int(total_frames - start))
    for i, offset in enumerate(range(min(num, int(total_frames - start)))):
        success, img = vidcap.read()

        with open(pose_files[i], 'r') as f:
          try:
            json_dat = json.load(f)['people'][0]
          except:
            #print ('no person on this frame, continue', vid)
            continue
        pose_kps = np.split(np.array(json_dat['pose_keypoints_2d']), 25)
        lhand_kps = np.split(np.array(json_dat['hand_left_keypoints_2d']), 21)
        rhand_kps = np.split(np.array(json_dat['hand_right_keypoints_2d']), 21)
        lhand_med = sum(lhand_kps)/len(lhand_kps)
        rhand_med = sum(rhand_kps)/len(rhand_kps)
        rel_poses = np.array([lhand_med[:2], rhand_med[:2], pose_kps[7][:2], \
            pose_kps[4][:2], pose_kps[6][:2], pose_kps[3][:2], pose_kps[0][:2]])

        h, w, c = img.shape
        if w < 226 or h < 226:
            d = 226. - min(w, h)
            sc = 1 + d / min(w, h)
            img = cv2.resize(img, dsize=(0, 0), fx=sc, fy=sc)
            rel_poses *= sc

        if w > 256 or h > 256:
            img = cv2.resize(img, (math.ceil(w * (256 / w)), math.ceil(h * (256 / h))))
            pose_sc = np.array([[(256. / w), (256. / h)]])
            rel_poses *= pose_sc

        img = (img / 255.) * 2 - 1
        frames.append(img)
        poses.append(rel_poses)
    return np.asarray(frames, dtype=np.float32), np.asarray(poses, dtype=np.float32)



def make_dataset(split_file, split, root, mode, num_classes, video_id=''):
    dataset = []
    with open(split_file, 'r') as f:
        data = json.load(f)

    i = 0
    count_skipping = 0
    pose_skipping = 0
    for vid in data.keys():
        if split == 'train':
            if data[vid]['subset'] not in ['train', 'val']:
                continue
        else:
            if data[vid]['subset'] != 'test':
                continue

        vid_root = root
        src = 0

        video_path = os.path.join(vid_root, vid + '.mp4')
        if not os.path.exists(video_path):
            continue

        num_frames = int(cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FRAME_COUNT))

        if mode == 'flow':
            num_frames = num_frames // 2

        pose_loc = os.path.join(vid_root.replace('body_crop_videos', 'body_crop_poses', 1), vid)
        if not os.path.exists(pose_loc) or not os.listdir(pose_loc):
          print ("Skip video due to pose", vid)
          pose_skipping += 1
          continue
        if num_frames - 0 < 9:
            print("Skip video ", vid)
            count_skipping += 1
            continue

        label = data[vid]['action'][0]

        #for l in range(num_frames):
        #    c_ = data[vid]['action'][0]
        #    label[c_][l] = 1

        dataset.append((vid, label, 0, num_frames, vid))

        i += 1
        #if i==10: break
    print("Skipped videos: ", count_skipping, "Pose skipped", pose_skipping)
    print(len(dataset))
    #import random
    #random.shuffle(dataset)
    return dataset




def get_num_class(split_file):
    classes = set()

    content = json.load(open(split_file))

    for vid in content.keys():
        class_id = content[vid]['action'][0]
        classes.add(class_id)

    return len(classes)


class PoseRgbTestDataset(data_utl.Dataset):
    """
    Purpose of this dataset is to load iamge squencenes and pose information for each test/eval video
    In this case, padding ans sampling diffres than PoseRgbDataset
    No frame sampling and padding, i.e use all the frames in a test video
    """
    def __init__(self, split_file, split, root, mode, transforms=None, video_id=None):
        self.num_classes = get_num_class(split_file)
        self.data = make_dataset(split_file, split, root, mode, self.num_classes, video_id=video_id)
        self.split_file = split_file
        self.transforms = transforms
        self.mode = mode
        self.root = root

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        vid, label, start_f, start_e, output_name = self.data[index]

        imgs, poses = load_rgb_frames_from_video(self.root, vid, start_f, start_e)
        imgs, poses = self.transforms((imgs, poses))
        ret_img = video_to_tensor(imgs)
        ret_img = video_to_tensor(imgs)
        ret_pose = torch.from_numpy(poses)
        return ret_img, ret_pose, label, vid

    def __len__(self):
        return len(self.data)
