# Rotate Conv Caffe
## Caffe
[![Build Status](https://travis-ci.org/BVLC/caffe.svg?branch=master)](https://travis-ci.org/BVLC/caffe)
[![License](https://img.shields.io/badge/license-BSD-blue.svg)](LICENSE)

Caffe is a deep learning framework made with expression, speed, and modularity in mind.
It is developed by the Berkeley Vision and Learning Center ([BVLC](http://bvlc.eecs.berkeley.edu)) and community contributors.

Check out the [project site](http://caffe.berkeleyvision.org) for all the details like

- [DIY Deep Learning for Vision with Caffe](https://docs.google.com/presentation/d/1UeKXVgRvvxg9OUdh_UiC5G71UMscNPlvArsWER41PsU/edit#slide=id.p)
- [Tutorial Documentation](http://caffe.berkeleyvision.org/tutorial/)
- [BVLC reference models](http://caffe.berkeleyvision.org/model_zoo.html) and the [community model zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo)
- [Installation instructions](http://caffe.berkeleyvision.org/installation.html)

and step-by-step examples.

[![Join the chat at https://gitter.im/BVLC/caffe](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/BVLC/caffe?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

Please join the [caffe-users group](https://groups.google.com/forum/#!forum/caffe-users) or [gitter chat](https://gitter.im/BVLC/caffe) to ask questions and talk about methods and models.
Framework development discussions and thorough bug reports are collected on [Issues](https://github.com/BVLC/caffe/issues).

Happy brewing!
## Installation
The installation is the same as caffe.

- [Installation instructions](http://caffe.berkeleyvision.org/installation.html)

## Rotate_convolution setup

This caffe could set convolution layers at rotate mode.

The file modified are listed here.

* /src/caffe/layers: 
```	
*.cpp *.cu
```
* /src/caffe/proto: 
```
caffe.proto
```
* /include/caffe/layers: 
```
*.hpp
```

Copy the modified file into corresponding caffe directory and make.

## rotation convolution layer
根据需要的旋转卷积层，在prototxt中输入

	rot_in:true
	
	rot_hide:true
	
	rot_out:true

类似于

	layer {
	  name: "conv1"
	  type: "Convolution"
	  bottom: "data"
	  top: "conv1"
	
	  convolution_param {
	    num_output: 96
	    pad: 2
	    kernel_size: 5
	    rot_in:true//设置为rot_in模式
	  }
	}
## rotation BN layer
输入```rot_mode:true```可以令BN层和scale层工作在rot mode.

注意训练的时候，将use_global_stats设置为false，否则BN层无法训练

另外注意由于caffe程序设计问题，scale层自动调用了biaslayer，因此我修改了biaslayer。如果使用biaslayer时bottom==top，则会自动进入rot模式。


	layer {
		bottom: "conv1"
		top: "conv1"
		name: "bn1"
		type: "BatchNorm"
		batch_norm_param {
			use_global_stats: false
			eps: 1e-4
			rot_mode:true//设置为rot模式
		}
	}
	
	layer {
		bottom: "conv1"
		top: "conv1"
		name: "scale1"
		type: "Scale"
		scale_param {
			bias_term: true
			rot_mode:true//设置为rot模式
		}
	}
## cross_entropy loss
由于caffe自带的cross_entropy loss前面加了sigmoid，并且要求输入label为一个数组。在用于单标签问题时有点麻烦。

该实现基于softmax，并不修改bp过程，只将loss更换为 cross_entropy loss。

使用如下：

	layer {
	  name: "loss"
	  type: "SoftmaxWithLoss"
	  bottom: "conv12"
	  bottom: "label"
	  top: "loss"
	  loss_param{cross_entropy:true}
	}
# Data aug & gcn
在data层加入transform_param，具体如下。

	transform_param{
	  gcn: true
	  affine: true
	  negate: true
	  do_flip: true
	  allow_stretch: true
	  zoom_min:0.909
	  zoom_max:1.1
	  translation_min: -5
	  translation_max: 5
	  rotation_min: 0
	  rotation_max: 360
	  shear_min: -20
	  shear_max: 20
	  downscale: 0.9
	  mean_value: 128
	  scale: 0.078125
	  input_size:95
	}