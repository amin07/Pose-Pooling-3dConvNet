import sys
import json
import math
import os
import os.path
import random

import cv2
import numpy as np
import torch
import torch.utils.data as data_utl
import glob

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
    pose_files = sorted(glob.glob(pose_path+'/*')) 
    vidcap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
    vidcap.set(cv2.CAP_PROP_POS_FRAMES, start)
    #print (total_frames, len(pose_files))
    pose_files = pose_files[start:]
    poses = []
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

        img = (img / 255.) * 2 - 1    # works better
        #img = (img / 255.)
        frames.append(img)
        poses.append(rel_poses)
    return np.asarray(frames, dtype=np.float32), np.asarray(poses, dtype=np.float32)


def make_dataset(split_file, split, root, mode, num_classes, class_id=None):
    dataset = []
    with open(split_file, 'r') as f:
        data = json.load(f)

    i = 0
    count_skipping = 0
    pose_skipping = 0
    for vid in data.keys():
        if split == 'train':
            #if data[vid]['subset'] not in ['train']:
            if data[vid]['subset'] not in ['train', 'val']:
                continue
        else:
            if data[vid]['subset'] != 'test':
                continue

        vid_root = root['word']
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

        label = np.zeros((num_classes, num_frames), np.float32)

        for l in range(num_frames):
            c_ = data[vid]['action'][0]
            label[c_][l] = 1
        if class_id!=None and class_id!=data[vid]['action'][0] : continue
        if len(vid) == 5:
            dataset.append((vid, label, src, 0, data[vid]['action'][2] - data[vid]['action'][1]))
        elif len(vid) == 6:  ## sign kws instances
            dataset.append((vid, label, src, data[vid]['action'][1], data[vid]['action'][2] - data[vid]['action'][1]))

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


class PoseRgbDataset(data_utl.Dataset):

    def __init__(self, split_file, split, root, mode, transforms=None, garbage_label=False, class_id=None):
        self.num_classes = get_num_class(split_file)

        self.data = make_dataset(split_file, split, root, mode, num_classes=self.num_classes, class_id=class_id)
        self.split_file = split_file
        self.transforms = transforms
        self.mode = mode
        self.root = root
        self.gl = garbage_label
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        vid, label, src, start_frame, nf = self.data[index]

        total_frames = 64

        try:
            start_f = random.randint(0, nf - total_frames - 1) + start_frame
        except ValueError:
            start_f = start_frame

        imgs, poses = load_rgb_frames_from_video(self.root['word'], vid, start_f, total_frames)
        imgs, poses, label = self.pad(imgs, poses, label, total_frames)
        imgs, poses = self.transforms((imgs, poses))
        #print (label.sum(axis=-1))
        if self.gl:
          label = self.fix_labels(label, poses)
        
        #print (label.sum(axis=-1))
        #sys.exit()
        ret_lab = torch.from_numpy(label)
        ret_img = video_to_tensor(imgs)
        ret_pose = torch.from_numpy(poses)
        """
        imgs = ret_img.permute(1, 2, 3, 0).numpy()
        poses = ret_pose.numpy()
        print (imgs.shape, poses.shape, vid)
        sid = range(0, imgs.shape[0], imgs.shape[0]//8)
        imgs, poses = imgs[sid], poses[sid]
        for i in range(imgs.shape[0]):
          for j in range(poses.shape[1]):
            joint_x, joint_y = int(poses[i][j][0]), int(poses[i][j][1])
             
            print (type(imgs[i]), imgs[i].dtype, imgs[i].shape, poses[i].shape, np.sum(imgs[i]))
            cv2.circle(imgs[i], (joint_x, joint_y), 6, (255,255,255), -1)      
          cv2.imshow('frame', imgs[i])
          key = cv2.waitKey(0)
          if key==27:
            cv2.destroyAllWindows()
            sys.exit()
        #sys.exit()
        #print (vid, imgs.shape, poses.shape)
                    
        ret_lab = torch.from_numpy(label)
        ret_img = video_to_tensor(imgs)
        ret_pose = torch.from_numpy(poses)
        """
        return ret_img, ret_pose, ret_lab, vid

    def fix_labels(self, label, poses):
      hand_pose = poses[:,2:4]      # taking both wrists
      hand_pose[hand_pose<=0.] = 0.
      hand_motion = hand_pose[0:1] - hand_pose
      hm = np.square(hand_motion).sum(axis=-1).sum(axis=-1)
      label[:, hm==0.] = 0.  # garbage
      #label[self.num_classes-1, hm==0.] = 1.  # garbage
      return label

    def __len__(self):
        return len(self.data)

    def pad(self, imgs, poses, label, total_frames):
        if imgs.shape[0] < total_frames:
            num_padding = total_frames - imgs.shape[0]

            if num_padding:
                prob = np.random.random_sample()
                if prob > 0.5:
                    pad_img = imgs[0]
                    pad_pose = poses[0]
                    pad = np.tile(np.expand_dims(pad_img, axis=0), (num_padding, 1, 1, 1))
                    padded_imgs = np.concatenate([imgs, pad], axis=0)
                    pad = np.tile(np.expand_dims(pad_pose, axis=0), (num_padding, 1, 1))
                    padded_pose = np.concatenate([poses, pad], axis=0)
                else:
                    pad_img = imgs[-1]
                    pad_pose = poses[-1]
                    pad = np.tile(np.expand_dims(pad_img, axis=0), (num_padding, 1, 1, 1))
                    padded_imgs = np.concatenate([imgs, pad], axis=0)
                    pad = np.tile(np.expand_dims(pad_pose, axis=0), (num_padding, 1, 1))
                    padded_pose = np.concatenate([poses, pad], axis=0)

        else:
            padded_imgs = imgs
            padded_pose = poses

        label = label[:, 0]
        label = np.tile(label, (total_frames, 1)).transpose((1, 0))

        return padded_imgs, padded_pose, label
