import collections
import glob
import math
import os
import argparse

import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from torchvision import transforms
import vposetransforms
import numpy as np

import torch.nn.functional as F
from pytorch_i3d import LayersPoseLocalI3d

from datasets.rgb_pose_dataset_test import PoseRgbTestDataset as Dataset
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('-data_split', type=str, default='test', help='data split using in evaluation. options : test or train')
parser.add_argument('-mode', type=str, help='rgb or flow')
parser.add_argument('-save_model', type=str)
parser.add_argument('-root', type=str)
parser.add_argument('-current_class', type=int, default=0)
parser.add_argument('-gpu', type=int, default=0)
parser.add_argument('-show-confusion', action='store_true')
parser.add_argument('-yield_fts', action='store_true')
parser.add_argument('-checkpoint', '-ckpt', required=True, type=str)
parser.add_argument('-end_point', type=str, default='', help='endpoint_current_run', required=True)
parser.add_argument('-logit_list', type=str, default='', help='comma separated lists of end point names, scores are fused from these end points')
parser.add_argument('-run_mode', type=str, default='', help='if blank normal test evaluation done, if save_logits then save logits from pt model')
parser.add_argument('-logit_loc', type=str, default='', help='in savelogit mode, give full location with specific folder for each model, in runlogits mode give only parent folder')

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '{}'.format(args.gpu)



def get_fused_logits(video_id, logit_loc, fusion='mean'):
  
  logits = []
  #print (logit_loc)
  for lts in logit_loc:
    lt = np.load(os.path.join(lts, video_id)+'.npy')
    logits.append(np.amax(lt, axis=-1, keepdims=True))
  #return torch.from_numpy(np.mean(np.stack([l for l in logits], axis=-1), axis=-1))
  return logits

def run_on_saved(init_lr=0.1,
        max_steps=64e3,
        mode='rgb',
        root='/ssd/Charades_v1_rgb',
        train_split='charades/charades.json',
        batch_size=3 * 15,
        save_model='',
        weights=None, 
        logit_loc=''):
    ''' logit location is contains several folders each of which contain a logit for a test video id '''
    assert args.logit_list!='', 'logit lists must be provided'
    # setup dataset
    test_transforms = transforms.Compose([vposetransforms.CenterCrop(224)])
      
    dataset_split = args.data_split
    assert dataset_split!='', "Dataset split is required, args.data_split is empty!"
    print ("running saved logits for ", dataset_split, "data")
    
    val_dataset = Dataset(train_split, dataset_split, root, mode, test_transforms)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1,
                                                 shuffle=True, num_workers=2,
                                                 pin_memory=False)

    dataloaders = {'test': val_dataloader}
    datasets = {'test': val_dataset}

    # setup the model

    correct = 0
    correct_5 = 0
    correct_10 = 0

    top1_fp = np.zeros(num_classes, dtype=np.int)
    top1_tp = np.zeros(num_classes, dtype=np.int)

    top5_fp = np.zeros(num_classes, dtype=np.int)
    top5_tp = np.zeros(num_classes, dtype=np.int)

    top10_fp = np.zeros(num_classes, dtype=np.int)
    top10_tp = np.zeros(num_classes, dtype=np.int)

    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int)
   
    logit_list = [lt.strip() for lt in args.logit_list.strip().split(",")]
    logit_loc = [os.path.join(args.logit_loc, lt) for lt in logit_list]
    logit_dict = collections.defaultdict(lambda : collections.defaultdict(lambda : []))
    label_dict = collections.defaultdict()
    for data in dataloaders["test"]:
        inputs, pose_inputs, labels, video_id = data  # inputs: b, c, t, h, w
        
        stacked_logits = get_fused_logits(video_id[0], logit_loc)
        #print (type(stacked_logits[0]), stacked_logits[0].shape)
        #sys.exit()
        for bri, br in enumerate(logit_list):
          logit_dict[br][video_id[0]] = stacked_logits[bri]
        label_dict[video_id[0]] = labels[0].item()
        # break
        per_frame_logits = torch.from_numpy(np.mean(np.stack(stacked_logits, axis=-1), axis=-1))
        predictions = torch.max(per_frame_logits, dim=2)[0]
        out_labels = np.argsort(predictions.cpu().detach().numpy()[0])
        out_probs = np.sort(predictions.cpu().detach().numpy()[0])
        confusion_matrix[labels[0].item(), out_labels[-1]] += 1        

        if labels[0].item() in out_labels[-5:]:
            correct_5 += 1
            top5_tp[labels[0].item()] += 1
        else:
            top5_fp[labels[0].item()] += 1
        if labels[0].item() in out_labels[-10:]:
            correct_10 += 1
            top10_tp[labels[0].item()] += 1
        else:
            top10_fp[labels[0].item()] += 1
        if torch.argmax(predictions[0]).item() == labels[0].item():
            correct += 1
            top1_tp[labels[0].item()] += 1
        else:
            top1_fp[labels[0].item()] += 1
        print(video_id, float(correct) / len(dataloaders["test"]), float(correct_5) / len(dataloaders["test"]),
              float(correct_10) / len(dataloaders["test"]))
        # per-class accuracy
    top1_per_class = np.mean(top1_tp / (top1_tp + top1_fp))
    top5_per_class = np.mean(top5_tp / (top5_tp + top5_fp))
    top10_per_class = np.mean(top10_tp / (top10_tp + top10_fp))
    print('top-k average per class acc: {}, {}, {}'.format(top1_per_class, top5_per_class, top10_per_class))
    if args.show_confusion:
      cplot.understand_conf(logit_dict, label_dict , class_file='preprocess/wlasl_class_list.txt')
      """
      model_name = "_".join(logit_list)
      print (model_name, 'confusion plot call')
      cplot.plot_conf_graph(confusion_matrix, class_file='preprocess/wlasl_class_list.txt', model_name=model_name) 
      for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
          if confusion_matrix[i, j] > 0.:
            print (i, j, confusion_matrix[i, j])
      """

def save_model_logits(init_lr=0.1,
        max_steps=64e3,
        mode='rgb',
        root='/ssd/Charades_v1_rgb',
        train_split='charades/charades.json',
        batch_size=3 * 15,
        save_model='',
        weights=None,
        logit_loc=''):
    assert logit_loc!='', "Please specify a logit loc!"
    # setup dataset
    test_transforms = transforms.Compose([vposetransforms.CenterCrop(224)])

    dataset_split = args.data_split
    assert dataset_split!='', "Dataset split is required, args.data_split is empty!"
    print ("saving logits for ", dataset_split, "data")
    val_dataset = Dataset(train_split, dataset_split, root, mode, test_transforms)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1,
                                                 shuffle=False, num_workers=2,
                                                 pin_memory=False)

    dataloaders = {'test': val_dataloader}
    datasets = {'test': val_dataset}

    
    # setup the model
    pl_i3d = LayersPoseLocalI3d(num_classes=num_classes, endpoints=[args.end_point])
    if weights:
      print ('Loading pretrained model', weights)
      pl_i3d.load_state_dict(torch.load(weights))
    pl_i3d.cuda()
    pl_i3d = nn.DataParallel(pl_i3d)

    pl_i3d.cuda()
    pl_i3d = nn.DataParallel(pl_i3d)
    pl_i3d.eval()

    correct = 0
    correct_5 = 0
    correct_10 = 0

    top1_fp = np.zeros(num_classes, dtype=np.int)
    top1_tp = np.zeros(num_classes, dtype=np.int)

    top5_fp = np.zeros(num_classes, dtype=np.int)
    top5_tp = np.zeros(num_classes, dtype=np.int)

    top10_fp = np.zeros(num_classes, dtype=np.int)
    top10_tp = np.zeros(num_classes, dtype=np.int)

    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int)
    if not os.path.exists(logit_loc):
      os.makedirs(logit_loc)
    
    yield_fts = args.yield_fts
    from tqdm import tqdm
    for data in tqdm(dataloaders["test"]):      # test can contain train data based on split input
      inputs, pose_inputs, labels, video_id = data  # inputs: b, c, t, h, w
      if yield_fts:
        per_frame_logits, per_frame_fts = pl_i3d(inputs, pose_inputs, yield_fts=yield_fts)
        print (per_frame_fts.size())
        np.save(os.path.join(logit_loc, video_id[0])+'.npy', per_frame_fts.detach().cpu().numpy())
      else:
        per_frame_logits = pl_i3d(inputs, pose_inputs)
        np.save(os.path.join(logit_loc, video_id[0])+'.npy', per_frame_logits.detach().cpu().numpy())
 
 
def run(init_lr=0.1,
        max_steps=64e3,
        mode='rgb',
        root='/ssd/Charades_v1_rgb',
        train_split='charades/charades.json',
        batch_size=3 * 15,
        save_model='',
        weights=None):
    # setup dataset
    test_transforms = transforms.Compose([vposetransforms.CenterCrop(224)])

    val_dataset = Dataset(train_split, 'test', root, mode, test_transforms)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1,
                                                 shuffle=False, num_workers=2,
                                                 pin_memory=False)

    dataloaders = {'test': val_dataloader}
    datasets = {'test': val_dataset}

    # setup the model, run
    pl_i3d = LayersPoseLocalI3d(num_classes=num_classes, endpoints=[args.end_point])
    if weights:
      print ('Loading pretrained model', weights)
      pl_i3d.load_state_dict(torch.load(weights))
    pl_i3d.cuda()
    pl_i3d = nn.DataParallel(pl_i3d)

    pl_i3d.cuda()
    pl_i3d = nn.DataParallel(pl_i3d)
    pl_i3d.eval()

    correct = 0
    correct_5 = 0
    correct_10 = 0

    top1_fp = np.zeros(num_classes, dtype=np.int)
    top1_tp = np.zeros(num_classes, dtype=np.int)

    top5_fp = np.zeros(num_classes, dtype=np.int)
    top5_tp = np.zeros(num_classes, dtype=np.int)

    top10_fp = np.zeros(num_classes, dtype=np.int)
    top10_tp = np.zeros(num_classes, dtype=np.int)

    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int)
    for data in dataloaders["test"]:
        inputs, pose_inputs, labels, video_id = data  # inputs: b, c, t, h, w
        inputs = inputs.cuda().float()
        pose_inputs = pose_inputs.cuda().float()
        per_frame_logits = pl_i3d(inputs, pose_inputs)

        predictions = torch.max(per_frame_logits, dim=2)[0]
        out_labels = np.argsort(predictions.cpu().detach().numpy()[0])
        out_probs = np.sort(predictions.cpu().detach().numpy()[0])
        confusion_matrix[labels[0].item(), out_labels[-1]] += 1        

        if labels[0].item() in out_labels[-5:]:
            correct_5 += 1
            top5_tp[labels[0].item()] += 1
        else:
            top5_fp[labels[0].item()] += 1
        if labels[0].item() in out_labels[-10:]:
            correct_10 += 1
            top10_tp[labels[0].item()] += 1
        else:
            top10_fp[labels[0].item()] += 1
        if torch.argmax(predictions[0]).item() == labels[0].item():
            correct += 1
            top1_tp[labels[0].item()] += 1
        else:
            top1_fp[labels[0].item()] += 1
        print(video_id, float(correct) / len(dataloaders["test"]), float(correct_5) / len(dataloaders["test"]),
              float(correct_10) / len(dataloaders["test"]))
        # per-class accuracy
    top1_per_class = np.mean(top1_tp / (top1_tp + top1_fp))
    top5_per_class = np.mean(top5_tp / (top5_tp + top5_fp))
    top10_per_class = np.mean(top10_tp / (top10_tp + top10_fp))
    print('top-k average per class acc: {}, {}, {}'.format(top1_per_class, top5_per_class, top10_per_class))
    return top1_per_class, top5_per_class, top10_per_class    
    
    if args.show_confusion:
      cplot.plot_conf_graph(confusion_matrix, class_file='preprocess/wlasl_class_list.txt') 
      for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
          if confusion_matrix[i, j] > 0.:
            print (i, j, confusion_matrix[i, j])



if __name__ == '__main__':
    

    mode = 'rgb'
    num_classes = 100
    save_model = './saved_checkpoints/'    # location from where we want to load checkpoints to test on

    root='/home/ahosain/workspace/WLASL/start_kit/all_videos/body_crop_videos/'

    train_split = 'preprocess/nslt_{}.json'.format(num_classes)
    weights = './saved_checkpoints/{}'.format(args.checkpoint)
    
    if args.run_mode=='save_logits':
      assert args.logit_loc!='', "Please specify location logits to be saved"
      save_model_logits(mode=mode, root=root, train_split=train_split, weights=weights, logit_loc=args.logit_loc)
    elif args.run_mode=='run_logits':
      assert args.logit_loc!='', "Please specify location logits to read from"
      run_on_saved(mode=mode, root=root, save_model=save_model, train_split=train_split, weights=weights, logit_loc=args.logit_loc)
    elif args.run_mode=='multi':
      all_weights = glob.glob('checkpoints/{}'.format(args.checkpoint))
      print (all_weights)
      res_dict = {}
      for w in all_weights:
        res_dict[os.path.basename(w)] = run(mode=mode, root=root, save_model=save_model, train_split=train_split, weights=w)
      print ('final results')
      for k, v in res_dict.items():
        print (k, v)  
    else:
      run(mode=mode, root=root, save_model=save_model, train_split=train_split, weights=weights)
