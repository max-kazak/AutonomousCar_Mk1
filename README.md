# Behavioral Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
In this project, I used deep neural networks and convolutional neural networks to clone driving behavior. To achieve reliable autonomous driving from the car I traine neural network that outputs a steering angle to an autonomous vehicle based on the images from the front camera.

The Writeup
---

### Code

My project includes the following files:

 - model.py containing the script to create and train the model 
 - drive.py for driving the car in autonomous mode 
 - model.h5 containing a trained convolution neural network 
 - README.md summarizing the results 
 
In addition _data_ folder contains data samples that I acquired through simulation. 

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```
python drive.py model.h5
```

The _model.py_ file contains the code for training and saving the convolution neural network and can be executed with 
```
python model.py
```

###  Data Collection & Augmentation

#### Image and telemetry capture

To gather training data I used Udacity simulator in manual mode. By controlling speed and turning angle with keyboard and mouse I tried to keep car within the road boundaries. Simulator automatically recorded images captured by three front facing cameras and turning angle.

For better generalization I also drove a few laps in opposite direction which gave me additional training data.

#### Augmentations

Even with several laps in forward and backward directions I could still get more training data for better training results by augmenting existing data. 

First, I used images from front left and front right cameras as sets captured by central camera not forgetting to add compensation to turning angle measurements. This also gave me data for more extreme and mild turning scenarious on the track. As the results my dataset trippled in size.

Secondly, I flipped every image horizontally and reversed turnging angle in the training data, giving me twice the data.

In the end, through augmentations I got six times the size of original data for training purposes.

To speed up training and avoid distractions on the images, I also cropped out top part of the images, where sky is, as it doesn't help with driving, and bottom part, where the car hood is, which doesn't change from image to image anyway.

### Solution Design

To design the final solution for this problem I tried using the simplest solution first and iteratively adding complexity and fixing problems later. 

At first I used simplest neural network with only one fully-connected layer that outputs one feature (turning angle) and passed flattened images as inputs. Such network could keep car on the road only on the straight road, but it was enough to check that data inputs could be processed without problems and that simulator can be driven using resulting neural network model. 

Next I replaced said network with the architecture copied from ["End to End Learning for Self-Driving Cars" by Nvidia research team](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf). 

After attempting to train new network I faced the problem where validation error is randomly decreasing and increasing. To eliminate this unstable behavior I increased size of sample batches and decreased learning rate, which seemed to have fixed the problem. 

However new network still couldn't drive on sharp turns and seemed as if it couldn't handle unexpected situations, which is a symptom of overfitting. 

Next I tried adding Dropout layers before each computational layers (convolutions and dense). 

I also tried to increase model accuracy by training longer with smaller learning rates. I did that by introducing exponential decay to learning rate, which automatically reduced learning rate as the training progressed. 

All of these modifications allowed the car to pass half of the loop with occasional driving on the edge on the road. But it seemed that model avoided predicting big turning angles. I suspect that's because I tried driving as smoothly as possible. To fix this I added data from side cameras as described in data augmentation section, which added significantly more data with bigger turning angles. 

Finally, model achieved driving car on its own thorugh the whole length of the loop, which can be seen in the [video](https://youtu.be/wZrne38rO3M).

### Network Architecture



The Project
---
The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior 
* Design, train and validate a model that predicts a steering angle from image data
* Use the model to drive the vehicle autonomously around the first track in the simulator. The vehicle should remain on the road for an entire loop around the track.

### Dependencies
This lab requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

The lab enviroment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.

The simulator can be downloaded from the [simulator repo](https://github.com/udacity/self-driving-car-sim).

## Details About Files In This Directory

### `drive.py`

Usage of `drive.py` requires you have saved the trained model as an h5 file, i.e. `model.h5`. See the [Keras documentation](https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model) for how to create this file using the following command:
```sh
model.save(filepath)
```

Once the model has been saved, it can be used with drive.py using this command:

```sh
python drive.py model.h5
```

The above command will load the trained model and use the model to make predictions on individual images in real-time and send the predicted angle back to the server via a websocket connection.

Note: There is known local system's setting issue with replacing "," with "." when using drive.py. When this happens it can make predicted steering values clipped to max/min values. If this occurs, a known fix for this is to add "export LANG=en_US.utf8" to the bashrc file.

#### Saving a video of the autonomous agent

```sh
python drive.py model.h5 run1
```

The fourth argument, `run1`, is the directory in which to save the images seen by the agent. If the directory already exists, it'll be overwritten.

```sh
ls run1

[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_424.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_451.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_477.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_528.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_573.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_618.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_697.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_723.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_749.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_817.jpg
...
```

The image file name is a timestamp of when the image was seen. This information is used by `video.py` to create a chronological video of the agent driving.

### `video.py`

```sh
python video.py run1
```

Creates a video based on images found in the `run1` directory. The name of the video will be the name of the directory followed by `'.mp4'`, so, in this case the video will be `run1.mp4`.

Optionally, one can specify the FPS (frames per second) of the video:

```sh
python video.py run1 --fps 48
```

Will run the video at 48 FPS. The default FPS is 60.

#### Why create a video

1. It's been noted the simulator might perform differently based on the hardware. So if your model drives succesfully on your machine it might not on another machine (your reviewer). Saving a video is a solid backup in case this happens.
2. You could slightly alter the code in `drive.py` and/or `video.py` to create a video of what your model sees after the image is processed (may be helpful for debugging).

### Tips
- Please keep in mind that training images are loaded in BGR colorspace using cv2 while drive.py load images in RGB to predict the steering angles.


