# Pose-Pooling-3dConvNet
Pose pooling 3d Convolutional Neural Network (I3D) for isolated sign word recognition from video. <br>
**Summary :** Sign video classifier (I3D) network uses localized feature maps extracted using pose locations of hands.

## Publication
```
@article{poseGuidedPoolingHosain,
  title={Hand Pose Guided 3d Pooling for American Sign Language Recognition},
  author={Hosain, Al Amin and Selvam, Panneer and Pathak, Parth and Rangwala, Huzefa and Kosecka, Jana},
  booktitle={In Proceedings of Winter Conference of Application in Computer Vision (WACV), 2021},
  year={2021}
}
```
## Requirements
Python, PyTorch

## Training the Network
1. Download and preprocess the data from https://github.com/dxli94/WLASL
2. Put the preprocessed data in a directory titled such as ```downloaded_data```
3. Set few variables in the ```__main__``` section of training script titled as ```train_network.py```
    - root : location where preprocessed data saved
    - save_model : location where checkpoints will be stored during the training process
    - train_split & configuration files : which subset of WLASL dataset used in training procedure
    - weights : this variables points to a pre-trained weight stored under ```weights``` directory
4. Run the training script simply using the command, ```python train_network.py -end_point <I3D Endpoint>```
5. The ```end_point``` option will take I3D network end point as value. Classifier will be trained using hand pose guided features from this branch.

## Evaluating the Network
