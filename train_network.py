import cv2
import os
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from torchvision import transforms
import vposetransforms
import numpy as np
from configs import Config
from pytorch_i3d import LayersPoseLocalI3d


from datasets.rgb_pose_dataset import PoseRgbDataset as Dataset
parser = argparse.ArgumentParser()
parser.add_argument('-mode', type=str, help='rgb or flow')
parser.add_argument('-save_model', type=str)
parser.add_argument('-root', type=str)
parser.add_argument('-num_class', type=int)
parser.add_argument('-class_id', type=int, default=None)
parser.add_argument('-gpu', type=str, default='0')
parser.add_argument('-model_tag', type=str, default='', help='identifier for saved model')
parser.add_argument('-pt_weight', type=str, default='', help='this weight for pretraining/fintuning a model')
parser.add_argument('-end_point', type=str, default='', help='endpoint_current_run', required=True)
parser.add_argument('-garbage_frame', action='store_true')

args = parser.parse_args()
print (args)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '{}'.format(args.gpu)


torch.manual_seed(0)
np.random.seed(0)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def run(configs,
        mode='rgb',
        root='/ssd/Charades_v1_rgb',
        train_split='charades/charades.json',
        save_model='',
        weights=None):
    print(configs)

    # setup dataset
    train_transforms = transforms.Compose([vposetransforms.RandomCrop(224), \
                                    vposetransforms.RandomHorizontalFlip()])
    test_transforms = transforms.Compose([vposetransforms.CenterCrop(224)])

    dataset = Dataset(train_split, 'train', root, mode, train_transforms, garbage_label=args.garbage_frame, class_id=args.class_id)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=configs.batch_size, shuffle=True, num_workers=0,
                                             pin_memory=True)

    val_dataset = Dataset(train_split, 'test', root, mode, test_transforms, garbage_label=args.garbage_frame, class_id=args.class_id)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=configs.batch_size, shuffle=True, num_workers=2,
                                                 pin_memory=False)

    dataloaders = {'train': dataloader, 'test': val_dataloader}
    datasets = {'train': dataset, 'test': val_dataset}

    num_classes = dataset.num_classes
    pl_i3d = LayersPoseLocalI3d(weights=weights, num_classes=num_classes, endpoints = [args.end_point])
    if args.pt_weight:
      print ("Weights loaded for pre training purpose", args.pt_weight)
      pl_i3d.load_state_dict(torch.load(args.pt_weight))

    pl_i3d.cuda()
    pl_i3d = nn.DataParallel(pl_i3d)

    lr = configs.init_lr
    weight_decay = configs.adam_weight_decay
    optimizer = optim.Adam(pl_i3d.parameters(), lr=lr, weight_decay=weight_decay)

    num_steps_per_update = configs.update_per_step  # accum gradient
    steps = 0
    epoch = 0

    best_val_score = 0
    # train it
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.3)
    while steps < configs.max_steps and epoch < 400:  # for epoch in range(num_epochs):
        print('Step {}/{}'.format(steps, configs.max_steps))
        print('-' * 10)

        epoch += 1
        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            collected_vids = []

            if phase == 'train':
                pl_i3d.train(True)
            else:
                pl_i3d.train(False)  # Set model to evaluate mode

            tot_loss = 0.0
            tot_loc_loss = 0.0
            tot_cls_loss = 0.0
            num_iter = 0
            optimizer.zero_grad()

            confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int)
            # Iterate over data.
            for data in dataloaders[phase]:
                num_iter += 1
                # get the inputs
                if data == -1: # bracewell does not compile opencv with ffmpeg, strange errors occur resulting in no video loaded
                    continue

                # inputs, labels, vid, src = data
                inputs, pose_inputs, labels, vid = data
                #print (inputs.size(), labels.size(), vid[0])
                #vis_sample_with_pose(inputs[0].permute(1,2,3,0).numpy(), pose_inputs[0].numpy())             
                #continue
                #inputs = inputs.cuda()
                #pose_inputs = pose_inputs.cuda()
                #pl_i3d(inputs, pose_inputs)
                #import sys
                #sys.exit()
                #continue
                #print (inputs.size(), pose_inputs.size())
                #import sys
                #sys.exit() 
                # wrap them in Variable
                inputs = inputs.cuda().float()
                pose_inputs = pose_inputs.cuda().float()
                t = inputs.size(2)
                labels = labels.cuda()

                per_frame_logits = pl_i3d(inputs, pose_inputs)
                #print (per_frame_logits.size())
                #import sys
                #sys.exit()
                # upsample to input size
                #AA upsample to last dim, last dim is batchxnum_cx7 for input temp 64
                per_frame_logits = F.interpolate(per_frame_logits, t, mode='linear')
                # compute localization loss
                loc_loss = F.binary_cross_entropy_with_logits(per_frame_logits, labels)
                tot_loc_loss += loc_loss.data.item()

                predictions = torch.max(per_frame_logits, dim=2)[0]
                gt = torch.max(labels, dim=2)[0]

                # compute classification loss (with max-pooling along time B x C x T)
                cls_loss = F.binary_cross_entropy_with_logits(torch.max(per_frame_logits, dim=2)[0],
                                                              torch.max(labels, dim=2)[0])
                tot_cls_loss += cls_loss.data.item()

                for i in range(per_frame_logits.shape[0]):
                    confusion_matrix[torch.argmax(gt[i]).item(), torch.argmax(predictions[i]).item()] += 1

                loss = (0.5 * loc_loss + 0.5 * cls_loss) / num_steps_per_update
                tot_loss += loss.data.item()
                if num_iter == num_steps_per_update // 2:
                    print(epoch, steps, loss.data.item())
                loss.backward()

                if num_iter == num_steps_per_update and phase == 'train':
                    steps += 1
                    num_iter = 0
                    optimizer.step()
                    optimizer.zero_grad()
                    # lr_sched.step()
                    if steps % 10 == 0:
                        acc = float(np.trace(confusion_matrix)) / np.sum(confusion_matrix)
                        print(
                            'Epoch {} {} Loc Loss: {:.4f} Cls Loss: {:.4f} Tot Loss: {:.4f} Accu :{:.4f}'.format(epoch,
                                                                                                                 phase,
                                                                                                                 tot_loc_loss / (10 * num_steps_per_update),
                                                                                                                 tot_cls_loss / (10 * num_steps_per_update),
                                                                                                                 tot_loss / 10,
                                                                                                                 acc), flush=True)
                        tot_loss = tot_loc_loss = tot_cls_loss = 0.
            if phase == 'test':
                val_score = float(np.trace(confusion_matrix)) / np.sum(confusion_matrix)
                if val_score > best_val_score or epoch % 2 == 0:
                    best_val_score = val_score
                    model_name = save_model + "ppc_" + str(num_classes) + "_" + str(steps).zfill(6) + '_%3f.pt' % val_score + '{}_{}'.format(args.model_tag, args.end_point)
                    torch.save(pl_i3d.module.state_dict(), model_name)
                    print(model_name)

                print('VALIDATION: {} Loc Loss: {:.4f} Cls Loss: {:.4f} Tot Loss: {:.4f} Accu :{:.4f}'.format(phase,
                                                                                                              tot_loc_loss / num_iter,
                                                                                                              tot_cls_loss / num_iter,
                                                                                                              (tot_loss * num_steps_per_update) / num_iter,
                                                                                                              val_score
                                                                                                              ))

                scheduler.step(tot_loss * num_steps_per_update / num_iter)


if __name__ == '__main__':
    # WLASL setting
    mode = 'rgb'
    #root = {'word': 'data/WLASL2000'}
    root = {'word': '/home/ahosain/workspace/WLASL/start_kit/all_videos/body_crop_videos/'}    ## NEED CHANGE 
    save_model = 'checkpoints/'
    train_split = 'preprocess/nslt_100.json'
    weights = '/home/ahosain/workspace/WLASL/i3d_wlsl/gmu_asl_weights/gmuasl_51_alamin_005938_0.911284.pt'    ## NEED to change
    config_file = 'configfiles/asl100.ini'
    configs = Config(config_file)
    print(root, train_split)
    run(configs=configs, mode=mode, root=root, save_model=save_model, train_split=train_split, weights=weights)
