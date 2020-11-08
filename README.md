# MVSPIN_NEW
This repo is for the paper:
**3D Human Pose and Shape Estimation Through Collaborative Learning and Multi-view Model-fitting**,
Zhongguo Li, Magnus Oskarsson, and Anders Heyden.

<p float="center">
  <img src="downtown_upstairs_00.gif" width="98%" />
</p>

# Installation and Fetch data
This repo is modified based on the method called [SPIN](https://github.com/nkolot/SPIN). If you want to run the code, you need to install depandencies and download the necessary data according to the installation instruction from the repo. 

# Demo
We provide a demo code for demonstrate the performance of our method. You can download the pre-trained model [here](https://drive.google.com/drive/folders/1kvpEyzXz8k5vhmLQnlLzQD7Qf-xvjY_T?usp=sharing). The person in the image should be in the center and cropped tightly. If you can provide the bounding box as the example in a pkl file, our demo still works. The bounding box is detected by CornerNet in our example.
Therefore, we provide two ways to run the demo.
1. Only test image in which the person is in the center and cropped tightly.
```
python demo_mvspin --trained_model=data/pre_trained.pt --test_image=examples/test.png
```
2. Test image with its correponding bounding box.
```
python demo_mvspin --trained_model=data/pre_trained.pt --test_image=examples/test.png --bbox=examples/test_bboxes.pkl
```
# Train
You can also train the network. Firstly, you need to download the training dataset [Human3.6M](http://vision.imar.ro/human3.6m/description.php) and [MPI-INF-3DHP](http://gvv.mpi-inf.mpg.de/3dhp-dataset/). 
For the two dataset, you can use the scripts in `datasets/preprocess` to generate npy files for training and testing. In these files, the information including image names, 2D/3D joint points and bounding boxes 
are set. Based on our pre-trained model, you can retrain the network:
```
python train.py --name train_example --pretrained_checkpoint=data/pre_trained.pt --run_mvsmplify
```
You can check the options for training in the script `utils/train_options.py`. After training the network, the trained models can be found in the foldes `logs/train_example`. You may use tensorboard to visulize the training process at the `logs`.

# Evaluatioin
We evaluate our method on three dataset:[Human3.6M](http://vision.imar.ro/human3.6m/description.php), [MPI-INF-3DHP](http://gvv.mpi-inf.mpg.de/3dhp-dataset/) and [3DPW](https://virtualhumans.mpi-inf.mpg.de/3DPW/).
You can download the npy files for evaluation or run the scrpits in `dataset/preprocess` to generate the files.
Then, you can evaluate the method on the three datasets.
```
python eval.py --checkpoint=data/pre_trained.pt --dataset=h36m-p1 --log_freq=20
```
You can change `--dataset` for different datasets. During the evaluation, you will see the results of Reconstruction Error and MPJPE. 

# Acknowledgement
We thank the [SPIN](https://github.com/nkolot/SPIN) sharing the code. And if you find our method is useful, please consider citing our paper
	@Inproceedings{Li2020MVSPIN,
	  Title          = {3D Human Pose and Shape Estimation Through Collaborative Learning and Multi-view Model-fitting},
	  Author         = {Zhongguo Li, Magnus Oskarsson, and Anders Heyden},
	  Booktitle      = {WACV},
	  Year           = {2021}
	}
