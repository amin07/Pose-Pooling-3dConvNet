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

from datasets.nslt_dataset_all import PoseRgbTestDataset as Dataset
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

def load_rgb_frames_from_video(video_path, start=0, num=-1):
    vidcap = cv2.VideoCapture(video_path)
    # vidcap = cv2.VideoCapture('/home/dxli/Desktop/dm_256.mp4')

    frames = []

    vidcap.set(cv2.CAP_PROP_POS_FRAMES, start)
    if num == -1:
        num = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

    for offset in range(num):
        success, img = vidcap.read()

        w, h, c = img.shape
        sc = 224 / w
        img = cv2.resize(img, dsize=(0, 0), fx=sc, fy=sc)

        img = (img / 255.) * 2 - 1

        frames.append(img)

    return torch.Tensor(np.asarray(frames, dtype=np.float32))


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


def ensemble(mode, root, train_split, weights, num_classes):
    # setup dataset
    test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])
    # test_transforms = transforms.Compose([])

    val_dataset = Dataset(train_split, 'test', root, mode, test_transforms)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1,
                                                 shuffle=False, num_workers=2,
                                                 pin_memory=False)

    dataloaders = {'test': val_dataloader}
    datasets = {'test': val_dataset}

    # setup the model
    if mode == 'flow':
        i3d = InceptionI3d(400, in_channels=2)
        i3d.load_state_dict(torch.load('weights/flow_imagenet.pt'))
    else:
        i3d = InceptionI3d(400, in_channels=3)
        i3d.load_state_dict(torch.load('weights/rgb_imagenet.pt'))
    i3d.replace_logits(num_classes)
    i3d.load_state_dict(torch.load(weights))  # nslt_2000_000700.pt nslt_1000_010800 nslt_300_005100.pt(best_results)  nslt_300_005500.pt(results_reported) nslt_2000_011400
    i3d.cuda()
    i3d = nn.DataParallel(i3d)
    i3d.eval()

    correct = 0
    correct_5 = 0
    correct_10 = 0
    # confusion_matrix = np.zeros((num_classes,num_classes), dtype=np.int)

    top1_fp = np.zeros(num_classes, dtype=np.int)
    top1_tp = np.zeros(num_classes, dtype=np.int)

    top5_fp = np.zeros(num_classes, dtype=np.int)
    top5_tp = np.zeros(num_classes, dtype=np.int)

    top10_fp = np.zeros(num_classes, dtype=np.int)
    top10_tp = np.zeros(num_classes, dtype=np.int)

    for data in dataloaders["test"]:
        inputs, labels, video_id = data  # inputs: b, c, t, h, w

        t = inputs.size(2)
        num = 64
        if t > num:
            num_segments = math.floor(t / num)

            segments = []
            for k in range(num_segments):
                segments.append(inputs[:, :, k*num: (k+1)*num, :, :])

            segments = torch.cat(segments, dim=0)
            per_frame_logits = i3d(segments)

            predictions = torch.mean(per_frame_logits, dim=2)

            if predictions.shape[0] > 1:
                predictions = torch.mean(predictions, dim=0)

        else:
            per_frame_logits = i3d(inputs)
            predictions = torch.mean(per_frame_logits, dim=2)[0]

        out_labels = np.argsort(predictions.cpu().detach().numpy())

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
        if torch.argmax(predictions).item() == labels[0].item():
            correct += 1
            top1_tp[labels[0].item()] += 1
        else:
            top1_fp[labels[0].item()] += 1
        print(video_id, float(correct) / len(dataloaders["test"]), float(correct_5) / len(dataloaders["test"]),
              float(correct_10) / len(dataloaders["test"]))

    top1_per_class = np.mean(top1_tp / (top1_tp + top1_fp))
    top5_per_class = np.mean(top5_tp / (top5_tp + top5_fp))
    top10_per_class = np.mean(top10_tp / (top10_tp + top10_fp))
    print('top-k average per class acc: {}, {}, {}'.format(top1_per_class, top5_per_class, top10_per_class))


def run_on_tensor(weights, ip_tensor, num_classes):
    i3d = InceptionI3d(400, in_channels=3)
    # i3d.load_state_dict(torch.load('models/rgb_imagenet.pt'))
    i3d.replace_logits(num_classes)
    i3d.load_state_dict(torch.load(weights))  # nslt_2000_000700.pt nslt_1000_010800 nslt_300_005100.pt(best_results)  nslt_300_005500.pt(results_reported) nslt_2000_011400
    i3d.cuda()
    i3d = nn.DataParallel(i3d)
    i3d.eval()
    t = ip_tensor.shape[2]
    ip_tensor.cuda()
    per_frame_logits = i3d(ip_tensor)
    predictions = F.upsample(per_frame_logits, t, mode='linear')
    predictions = predictions.transpose(2, 1)
    out_labels = np.argsort(predictions.cpu().detach().numpy()[0])
    plt.plot(range(len(arr)), F.softmax(torch.from_numpy(arr), dim=0).numpy())
    plt.show()
    return out_labels

def viz_frame_preds(weights, ip_tensor, num_classes):
    """ shows prediction of each frames with image  
        to see how consistent preds are in a video """
    i3d = InceptionI3d(400, in_channels=3)
    # i3d.load_state_dict(torch.load('models/rgb_imagenet.pt'))

    i3d.replace_logits(num_classes)
    i3d.load_state_dict(torch.load(weights))  # nslt_2000_000700.pt nslt_1000_010800 nslt_300_005100.pt(best_results)  nslt_300_005500.pt(results_reported) nslt_2000_011400
    i3d.cuda()
    i3d = nn.DataParallel(i3d)
    i3d.eval()

    t = ip_tensor.shape[2]
    ip_tensor.cuda()
    per_frame_logits = i3d(ip_tensor)
    print (per_frame_logits.size())
    predictions = F.upsample(per_frame_logits, t, mode='linear')
    sample_f = lambda m, n: [i*n//m + n//(2*m) for i in range(m)]
    predictions = predictions.transpose(2, 1)
    out_labels = np.argsort(predictions.cpu().detach().numpy()[0])
    out_im = ip_tensor[0].permute(1, 2, 3, 0).cpu().detach().numpy()
    out_im = [im for im in np.array(((out_im+1.)/2.)*255., dtype=np.uint8)]  ## reverse norm, see dataset file
    #print (out_im.shape)
    sample_ids = sample_f(per_frame_logits.size(2), t)
    #out_im = [out_im[i] for i in sample_ids]
    hstacks_im, vstack_im = [], []
    for j, i in enumerate(sample_ids):
      #print (out_im[i].shape, out_im[i][0,0], out_im[i].dtype)
      out_im[i] = cv2.cvtColor(cv2.cvtColor(out_im[i], cv2.COLOR_BGR2RGB), cv2.COLOR_BGR2RGB)
      #print (out_im[i].shape, out_im[i][0,0], out_im[i].dtype)
      #print (out_im[i])
      max_ind = out_labels[i][-1]
      max_logits = predictions.cpu().detach().numpy()[0][i][max_ind]
      out_im[i] = cv2.putText(out_im[i], str(max_ind)+str(', {:.2f}'.format(max_logits)), (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (225, 0, 0), 2)
      if hstacks_im and j%6==0:
        vstack_im.append(np.hstack(hstacks_im)) 
        hstacks_im.clear()
      
      hstacks_im.append(out_im[i])
    hstacks_im = hstacks_im + [np.zeros((224, 224, 3), dtype=np.uint8) for _ in range(6-len(hstacks_im))]
    vstack_im.append(np.hstack(hstacks_im))
    hstacks_im.clear()
    print ([v.shape for v in vstack_im])
    out_im = np.vstack(vstack_im)
    #print (out_im.shape)
    cv2.imshow('frame', out_im)
    #k = cv2.waitKey(-1)
    #if k==27:
    #  cv2.destroyAllWindows()
    #import sys
    #sys.exit()

    return out_labels



def get_slide_windows(frames, window_size, stride=1):
    indices = torch.arange(0, frames.shape[0])
    window_indices = indices.unfold(0, window_size, stride)

    return frames[window_indices, :, :, :].transpose(1, 2)



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
