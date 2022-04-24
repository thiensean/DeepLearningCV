# Concepts of Advanced Driving Assistance System
#### Deep Learning in Computer Vision / CNN / Image Classification / Object Localization / Mask R-CNN / Fast R-CNN / Tensorflow Keras / GradCAM / Tensorflow Functional API / Flask API / Streamlit
---

### Lim Thien Sean

#### Industry Mentors: <br>
Marie Stephen Leo:
https://www.linkedin.com/in/marie-stephen-leo <br>
https://stephen-leo.medium.com/ <br>

Shilpa Sindhe:
https://www.linkedin.com/in/shilpa-sindhe <br>
https://shilpa-leo.medium.com/ <br>

---
# Part 1 - Foundation of Classification & Localization

## Executive Summary

Introduction to ADAS:

<img src="./imgs/advancedadas.jpg" width=800 height=420 /> <br>
<i> Figure 1: Actual ADAS [18] </i>

<img src="./imgs/adas.JPG" width=800 height=420 /> <br>
<i> Figure 1-a: Various Tasks of ADAS [18] </i>

Fatalities caused by driving a car is statistically higher than flying, with more than 5 million accidents on the road compared to flying [1].

The automotive industry have been making significant advances that have been both impactful and significant to reducing accidents. With advances in deep learning in computer vision, LiDAR sensors, ToF cameras, the industry was able to move towards implementing advanced driver assistance as a precursor to fully self driving vehicles.

Deployment of computer vision in automotive vehicles have a set of unique requirements. They are required to 'see' like humans, classify pedestrians, street signs, vehicles, road hazards and localize these objects in their field of view. A further set of sensors like LiDAR and stereo cameras add a 3rd layer of 3D depth to the vast sensing arrays deployed in automotive vehicles.

<img src="./imgs/adas1.JPG" width=923 height=547 /> <br>
<i> Figure 1-b: Various Tasks of ADAS [25] </i>

Hence, advancements in deep learning in Computer Vision (CV) plays a huge role in advancing automotive safety. There are 3 topics involved (and more that were not researched heavily in this capstone): <br>
    1. Object classification <br>
    2. Object localization <br>
    3. Image segmentation <br>

These broad topics are widely researched and requires both speed and accuracy to be able to deployed reliably. Any lack of detection at any moment even for a split second due to poor scoring or object being obscured can cause the agenda of self-driving to fail.

Traffic signs classification & localization is a topic where automotive deployed computer vision application has value owing to traffic signs containing critical information about: <br>
    1. Stop signs - Mandatory stop <br>
    2. Entrance to highway - Build up speed for merging onto highway <br>
    3. Pedestrian crossing - Slow down where pedestrians are expected <br>
    4. Speed signs - Speed limits <br>
    etc. <br>
    
<img src="./imgs/classlocalize/63_63.png" width=600 height=400 /> <br>
<i> Figure 2: Example of object classification and Localization </i>

In part 1 of this project, the foundation of object classification and object localization will be explored. At the end of Part 1 of this project: <br>
    1. The model should be able to classify the image <br>
    2. The model should be able to localize the object <br>
    3. YOLOv4 which is a fast implementation of object classification & detection with a high frame rate will be tested on OpenCV, trajectory tracking will be used to link up object localization with object tracking. <br><br>

The dataset was obtained from 2 sources, 1 of which was from an online source with its source github listed in the source below. The 2nd source is from Google Street View. There is no intention of copyright infringement, all images are for education & research purposes with no intention for commercial deployment.

---

### Research Work

Object localization was a task with different solutions. A few solutions were researched to accomplish object localization:

1. Fast R-CNN
2. R-CNN
3. Single Shot Detector (SSD)
4. YOLO (You Only Look Once)

#### R-CNN

R-CNN is significantly slower than Fast R-CNN with a higher training time as this model architecture means that the model is classifying multiple region proposals per image [2].

It takes a great amount of time to return an image localization vertices due to its architecture of: <br> Region proposal regeneration -> Feature extraction -> Classification.

<img src="./imgs/models/regionproposal.JPG" width=330 height=300 align='left'/> <br><br><br><br><br><br><br><br><br><br><br><br><br><br>
<i> Figure 3: Region Proposal [3][14]</i>

Every warped region proposal has to be evaluated with ConvNet and this takes up computational power. [14] <br>

<img src="./imgs/models/comparisonmodels.png" width=800 height=350 /> <br>
<i> Figure 4: Comparison Between Models </i>

#### Fast R-CNN

Fast R-CNN works with an architecture of region proposal network.

It uses a CNN to extract feature maps from the image and in parallel, it tries to find region proposals.

Owing to the architecture, the features map had already been completed in the first step (CNN) and region proposals can now be mapped to the features map.

Fast R-CNN has an ROI pooling layer where there will be class scores for each region proposal. convNet, when comparing to R-CNN is only run once on an image hence it is much faster than R-CNN. <br>

<img src="./imgs/models/frcnn.png" width=800 height=520 /> <br>
<i> Figure 5: Fast R-CNN architecture [3]</i>

---

### Functional API with Classification and Regression predictions

#### Activation Functions

For this project, the Functional API will be used to build a model based off Fast R-CNN multi-prediction classification and regression outputs. <br><br>
A functional model allows the model to make data driven decisions while performing regression on the bounding box coordinates. This is done by connecting the classification and regression branches through the base layer, and this allows the model to understand where certain features lies using the feature maps that were created by the convolution layers feature extraction phase, and hence where to place the bounding boxes.
It connects the feature maps and the bounding box regression.

There will be 2 activation functions:

1. Softmax; Giving class probabilities
2. Sigmoid; For regression problem (Bounding Box)

---

### Dataset Preparation

There are no labels in the images in the dataset and the images have to be prepared by manually labeling.

LabelImg is a great software authored by: tzutalin [15] and it helps with labeling the image quickly by allowing user to draw bounding boxes and write bounding box vertices to an XML file in PascalVOC format so it can be parsed afterwards.

<img src="./imgs/labelimg.JPG" width=500 height=400 /> <br>
<i> Figure 6: LabelImg [3]</i>


It gives class, coordinates and file path upon saving.

<img src="./imgs/labelimg2.JPG" width=500 height=470 /> <br>
<i> Figure 7: LabelImg Bounding Box vertices on XML [3]</i>

### Modelling

Keras, a high level API to Tensorflow, was used to build Convolutional Neural Networks (CNNs) and trained on the traffic signs dataset. The classification problem was explored with a sequential model, followed by a functional model to achieve vertices output for object localization.

For the sequential model, the output expected was the class of the image which is identified through dictionary defined at the start of the codebook. The problem was defined to be a multi-class classification of 32 classes.

For the functional model, the images contained higher number of inputs, containing the image array and array containing 4 vertices of the boundary box labelled by human labeller (author of this project). The base layer were shared by both the classification branch and the localizer branch.

In both of the models, it is assumed that both images contained only one traffic sign.

Going back to the fundamentals, both models will have the base layers built with custom layers.

---
#### Sequential Model

In the Sequential model, there will be 2 Conv2D before a MaxPool2D layer to extract features.
A BatchNormalization layer was added to reduce overfitting and used to prevent exploding gradient issue - which means that large error gradients build up due to multiplications of large derivatives, causing large updates to neural network model weights. This causes the model to fail to learn. [9][10]

Dropout was added to prevent the model from becoming weight dependent by randomly dropping weights in layers. [9]

The activation function used for classification was 'Softmax' which will assign class probabilities to each class. [11]

<img src="./imgs/models/sequential.JPG" width=900 height=550 /> <br>
<i> Figure 8: Sequential Model </i>

---
#### Functional Model

The goal of the functional model is to build upon the model's capability to classify and return one more set of arrays, the 4 vertices predicting the bounding box containing the object.

Transitioning from Sequential to Functional model, the image array, class and the object bounding box vertices were added.

There were now 2 branches: cl_head and bb_head

'cl_head' contains classification labels. <br>
'bb_head' contains bounding box 4 vertices. <br>

The activation functions were different for classification branch and the localizer branch.

'Softmax' remained in use for classification as it predicts class probablities for each class.
'Sigmoid' was used for bounding box regression.

<img src="./imgs/models/Functional2.JPG" width=1100 height=550 /> <br>
Figure 9: Functional Model with Fast R-CNN architecture <br><br><br>

<img src="./imgs/models/model.png" width=350 height=1100 /> <br>
Figure 9-1: Visualizing Functional API model connections & output shape <br>

---
#### GradCAM

Gradient-weighted Class Activation Mapping (Grad-CAM) emphasizes image using the gradient of an image used for classification that flows into the final convolution layer. It produces a coarse localization heatmap that highlights the crucial regions that are being used to predict the classification. [12]

Using the gradients or the model loss, we can tell how the characteristics of the image are emphasized through the creation of heatmaps. [13]

In this project, the concept of Grad-CAM was applied to certain classification images.

From the images as follows: We can see how certain features are localized and emphasized.

---
In this example and in similar images, the model recognizes the 2 arrows.

<img src="./imgs/arrow1.png" width=200 height=200 align='left'/> <br><br><br><br><br><br><br><br><br><br>
<i> Figure 10: GradCAM 1 </i>

In this example and in similar images, the model recognizes the line across the image.

<img src="./imgs/gradcammask1.png" width=200 height=200 align='left'/> <br><br><br><br><br><br><br><br><br><br>
<i> Figure 11: GradCAM 2 </i>

In this example and in similar images, the model managed to localize and recognize the circle and its features.

<img src="./imgs/gradcammask2.png" width=200 height=200 align='left'/> <br><br><br><br><br><br><br><br><br><br>
<i> Figure 12: GradCAM 3 </i>

In this example and in similar images, the model recognizes and emphasizes the half split characteristics of this sign.

<img src="./imgs/gradcammask7.png" width=200 height=200 align='left'/> <br><br><br><br><br><br><br><br><br><br>
<i> Figure 13: GradCAM 4 </i>

### Loss Function

The loss function used for this project was Mean Squared Error (MSE) as it tells the model the error from the groundtruth.

Ideally, the loss function should be custom written as IoU loss function.

<img src="./imgs/loss.JPG" width=274.4 height=314.4 align='left'/> <br><br><br><br><br><br><br><br><br><br><br><br><br><br>
<i> Figure 14: Loss Function - Mean Squared Error [c] </i>

### Metric

There are 2 sets of metrics for classification and localization.

For classification, 2 useful metrics will be precision and AUC-ROC score.
These 2 metrics will substantiate the classification by telling us the precision from classification and the separability of the classification.

Intersection over Union (IoU) or 'Jaccard Distance' is a good measure of the accuracy of the regression. [b]

<img src="./imgs/IOU1.JPG" width=280 height=280 align='left' /> <br><br><br><br><br><br><br><br><br><br><br><br><br>
<i> Figure 15: IOU metric [b] </i>

#### Results of Classification

Classification results had an average precision score of <b>0.97</b>.

<img src="./imgs/precision.JPG" width=450 height=800 /> <br>
<i> Figure 16: Results of Classification [b] </i>

#### Results of object localization

Average results of IOU is <b>0.88</b>.

Below are samples of the images that were both classified and localized.

<img src="./imgs/classlocalize/10_10.png" width=500 height=350 /> <br>
<i> Figure 17: Image localization IOU results </i>

<img src="./imgs/classlocalize/12_12.png" width=500 height=350 /> <br>
<i> Figure 18: Image localization IOU results </i>

<img src="./imgs/classlocalize/55_55.png" width=500 height=350 /> <br>
<i> Figure 19: Image localization IOU results </i>

<img src="./imgs/classlocalize/78_78.png" width=500 height=350 /> <br>
<i> Figure 20: Image localization IOU results </i>

<img src="./imgs/classlocalize/62_62.png" width=500 height=350 /> <br>
<i> Figure 21: Image localization IOU results </i>

<img src="./imgs/classlocalize/70_70.png" width=500 height=350 /> <br>
<i> Figure 22: Image localization IOU results </i>

### Flask API

To enable fast rendering, the API would only return class and boundary box vertices.

The rendering of the bounding boxes would be performed on the local machine.

The file to serve the API was saved as 'serve.py' inside the working folder.

<img src="./imgs/api.JPG" width=950 height=500 /> <br>
<i> Figure 23: API structure </i>

### Simple web deployment

The model was deployed as a simple web app in Streamlit.

<img src="./imgs/streamlit.gif" width=1100 height=600 /> <br>
<i> Figure 24: Simple web app using Streamlit to demonstrate model deployment </i>

## Object Tracking

With the knowledge that YOLO has the fastest detection and the highest frame rate, yoloV4 weights was used for object tracking in video.
Using YOLOv4 weights, the weights were input onto OpenCV with DNN module to attempt to get the highest frame rate so that object trajectory can be coded.

Using YOLOv4 object detection, the coordinates of the center of each bounding box can be tracked. Each object detected was given an object ID.

By tracking the location of one frame to the next frame, the trajectory of the object can be tracked so each object in the frame has its own unique ID.

A threshold of 30 pixels were set so that if the object detection from the previous to the next frame exceed 30 pixels, the object would be popped.

Below are a sample of the results:

<img src="./imgs/opencvdnn.JPG" width=1150 height=750 /> <br>
<i> Figure 25: Trajectory tracking using yoloV4 with OpenCV </i>

This is a work in progress and might work better using image segmentation techniques. [17][m, Image Segmentation and Pattern Matching]

<img src="./imgs/trajectorytracking.gif" width=500 height=340 /> <br>
<i> Figure 26: Trimmed video with carplates censored. [18] </i>

---

## Part 1 - Conclusion

In conclusion, from this project, it can be seen that the objectives intended were achieved.
The model was built without using transfer learning from pretrained models. And this solidified the foundation of understanding architectures for multi-predictions problems.

This may not be the best method, but it is highly adaptable and can be tuned to different sets of images by using libraries like Grad-CAM to understand what each convolution layer sees eg. Lines, Contours, Edges.

In part 2 of this project, this project will research on:
1. Architectures like Single Shot Detectors, and it will solidify the aim of speed and accuracy.
2. Mask R-CNN, to compare results between simple convolutional network vs deep residual neural networks like ResNet101.

#### Room for improvements

It might be a good idea to replace the base layer with a pretrained model like mobilenetV3 / EfficientNet / ResNet / VGG16.

By making use of a pretrained model, it could improve results and reduce overfitting.

The image dataset could also be more zoomed out so that the model is able to see higher variations of the image compared to zoomed in images.

---

# Part 2 - Mask R-CNN with Deep Neural Networks (ResNet)

## Alternative Segmentation Architectures to be Studied

1. Mask R-CNN
2. SOLOv2
3. Mask-Former
4. Temporally Efficient Vision Transformer (TeViT) for VIS (new SOTA for VIS at ~70FPS)
5. STEGO (unsupervised segmentation)

## Mask R-CNN

Mask R-CNN is planned for implementation in this project. The Mask R-CNN architecture used was cloned from Matterport's Github [19].
To start off, Mask R-CNN with pre-trained COCO-weights were loaded. In this implementation, only the "heads" layer will be used to train the model.

Mask R-CNN was the state-of-the-art model used for image segmentation when it was first released, followed closely by new techniques on researches using transformers with Computer Vision, eg. Mask-Former.

The architecture of Mask R-CNN builds upon Faster-RCNN, hence follows a region based convolutional network that returns bounding box and its class label with confidence score. Mask R-CNN will be used to classify, localize and mask the traffic objects in this project.

<img src="./imgs/mask-rcnn-architecture.png" width=1150 height=450 /> <br>
<i> Figure 27: Mask R-CNN architecture [g] </i>

### Backbone - ResNet101

The Mask R-CNN to be used for this project has a backbone using ResNet101 [h] that uses a residual network that network architect use to allow deep neural networks while minimizing gradient instability or exploding/vanishing gradients.

Deeper neural networks using residual networks with a higher depth was proven to demonstrate higher accuracy in the paper "Deep Residual Learning for Image Recognition". [h]

ResNet allows very deep networks by learning the residual representation functions instead of learning the signal representation directly. [21]

One unique characteristic of ResNet vs plain networks is that it uses skip connections, by adding a skip connection to add the input to the output after multiple weight layers. [21]

A very effective way of visualizing this example was given in article "Review: ResNet — Winner of ILSVRC 2015 (Image Classification, Localization, Detection), by Dr Sik-Ho Tsang [21]. Where the weight layers in between the input x and output are represented by F(x), the resulting output H(x) is given by H(x) = F(x) + x.

Hence, the residual mapping to be learnt can be represented by F(x) = H(x) - x, where x represents the identity x.
Below is an excellent visualization in the article shared above. <br>

<img src="./imgs/skip.png" width=447 height=237 /> <br>
<i> Figure 28: Skip Connections in ResNet [21] </i> <br>

### Vanishing/Exploding gradients

To address why ResNet is suited to tackle the problem of vanishing/exploding gradients that comes with deep networks, firstly, we drill down the problem.

Activation functions such as sigmoid activation normalizes the input to a value of 0 to 1. In its simplest terms, its derivative represents the rate of change of the underlying term, the input that was normalized.

Hence, when a large change occurs in the input, a resulting large derivative value occurs but becomes very small when there is no resulting change afterwards.

This mathematical phenomenon is represented by the graph below:

<img src="./imgs/sigmoid.png" width=700 height=268 /> <br>
<i> Figure 29: Small derivative term after a large resulting change in input [22] </i>

#### Backpropagation

Neural network uses backpropagation to update its weights by using the chain rule to find the derivatives of the network by "backward propagation of errors" from the last layer to the first layer.

Small derivatives are multipled together when hidden layers uses activation functions like the sigmoid function [22]. This causes a problem because when the derivations are small, and these gets multipled, it gets exponentially smaller hence, vanishing gradient.

Similarly, in a deep plain network, when large derivatives get multipled, the gradient gets exponentially higher and hence, explodes.

When this phenomenon of vanishing/exploding gradient happens the weights and biases does not get updated effectively and the model does not learn properly.

#### Region Proposal Networks [p]

A Region Proposal Network (RPN) takes an image as input and outputs object proposals in terms of 'Bounding Box' by scoring each proposal based on its class.

The method to generating these region proposals can be visualized on a high level by sliding a network over the convolutional feature map output. This sliding window are mapped to a low dimension feature and then fed into 2 fully connected layers branching out into a box regression layer and a box classification layer.

This sliding network is repeated and shared with all locations in an image.

<img src="./imgs/rpn.JPG" width=853 height=526 /> <br>
<i> Figure 30: Region Proposal Network [p] </i>

The red box centered at the sliding window in the image above is referred to as the anchor.
An anchor has a scale and aspect ratio which by default is where k = 3 x 3.
          
The detections using a Region Proposal Network is a bounding box, class and its probabilities. 
          
Non-Max Suppression is used to reduce redundancy based on their class scores. The top overlapping proposals are collapsed and used for detection.

#### Losses for Training RPN

The loss function of this architecture for the bounding box and class works by assigning a binary class label (positive or negative) to each anchor.

This effectively recognizes each anchor as an object or not.

##### What defines a positive label?

1. If the intersection over Union (IoU) overlap with the ground truth is over the threshold of 0.7 (by default).
2. Anchors with the highest Intersection-over-Union (IoU) overlap.

##### What defines a negative label?

1. When the Intersection over Union (IoU) with the ground truth box is lower than 0.3.

With condition being <0.3 and >0.7, there will be some achors that are neither positive or negative and these do not contribute to the training objective. [p]

### Mask Branch [q]

In Mask-RCNN, it uses a Fully Convolution Network (FCN) for the pixel-to-pixel task.
This extends Faster R-CNN by adding a parallel branch for predicting segmentation masks on top of Faster R-CNN's ability to predict class and bounding box.

<img src="./imgs/mrcnn-branch.png" width=850 height=334 /> <br>
<i> Figure 31: Mask R-CNN Branch Extended from Faster R-CNN </i>

In the architecture explained in the original paper, the ResNet-FPN variant was adopted.

To produce a relatively high resolution output to achieve localization accuracy, the keypoint head consisted of a stack of 8, '3×3', '512-d' conv layers, followed by a deconv layer and 2× bilinear upscaling.

#### Pixel-to-Pixel Alignment

To allow pixel level alignment between network inputs and outputs, in the original paper, a simple and quantization-free layer called <b>RoiAlign</b> to preserve the exact spatial locations was proposed.

As a result of 'RoiAlign', the mask accuracy was improved by 10% to 50% [q].

<img src="./imgs/roialign.JPG" width=684 height=549 /> <br>
<i> Figure 32: RoiALign for Mask Alignment Improvement </i>

#### Instance Segmentation

In order to effectively achieve instance segmentation, the masking was decoupled from the class predictions.

The RoI classification branch was used to predict the category instead.

### Possible improvement to Mask R-CNN
A possible improvement to this project can be replacing the backbone from ResNet101 to a deeper or more efficient architecture.

There are many considerations in this scenario.

Below, is a comparison of architectures that was publicly released to 2019 since ResNet was first published.


---
## Modelling Setup

Matterport's Mask-RCNN has some differences from the official paper, it resizes but keeps the original image aspect ratio by padding the image. [19]

Mask R-CNN has a region proposal network that simultaneously predicts object bounds and objectness scores at different positions of the image. It also includes a Non-Maximum Suppression (NMS) post processing step for collapsing the ROI's into the smallest box that encapsulates the pixels of the object, using confidence intervals that are above threshold setup during initial configuration stage. [19]

### Data and Model Preparation

In this project, the setup is as follows: <br>
DETECTION_MIN_CONFIDENCE       0.9 <br>
DETECTION_NMS_THRESHOLD        0.3 <br>

The masks over the objects were annotated using "makesense.ai" [20] using polygons saved in JSON format.

<img src="./imgs/cars1.JPG" width=1150 height=710 /> <br>
<i> Figure 33: "Cars" Class Mask Annotation </i>

### Image Augmentation

Image augmentation was used to allow the model to see more variations of the same image. This was done using "imgaug", with parameters of the augmentation modified in python file "model.py".

### Callbacks

2 Callbacks were used for mask RCNN training:

1. Early Stopping
2. ReduceLROnPlateau with minimum learning rate set at 0.001.

'val_loss' was used for monitoring the training process to achieve the best results.

---

Mask R-CNN annotations and development ongoing..

---

# Part 3 - LiDAR

## LiDAR Point Cloud Segmentation

<img src="./imgs/lidarsemanticsegment.png" width=560 height=420 /> <br>
<i> Figure 34: LiDAR point cloud [26]</i>

LiDAR point clouds can be segmented by unsupervised machine learning techniques like DBSCAN.

They can be treated as 'pixels' with depth information, adding another dimension to the point cloud visualization.

### DBSCAN

DBSCAN, Density-Based Spatial Clustering of Applications with Noise, an unsupervised machine learning algorithm is useful in this aspect. LiDAR point clouds data returns unlabeled and heavily clustered when the LiDAR laser beam lands and is reflected upon an object.

LiDAR data characteristic makes it suitable to use DBSCAN or K-means clustering Machine Learning techniques to segment it.
Using this method, the algorithm can be optimized and noise points ignored if they are not close enough to core points to be considered as part of the cluster, or belonging to the border point.

<img src="./imgs/lidar2.png" width=1400 height=564 /> <br>
<i> Figure 35: LiDAR Point Cloud Segmentation [28] </i>

This makes it very useful for LiDAR since LiDAR point cloud density and the rate of scanning (for MEMs LiDAR) is known beforehand. This allows the engineer to tune the 'Epsilon' value of DBSCAN to segment the point cloud effectively. 

The epsilon value can be optimized and can be studied further. [n]

### K-means clustering

K-means clustering, an unsupervised machine learning clustering technique works by minimizing the sum of the distances between each object or cluster centroid by predefining the 'K' value, also known as the the number of centroids. This algorithm can also be used to segment the point cloud but its not as intuitive as DBSCAN due to the need to define the number of centroids.

### LiDAR Point Cloud Segmentation with Bounding Box

<img src="./imgs/lidar3.png" width=1400 height=555 /> <br>
<i> Figure 36: LiDAR Point Cloud Segmentation with Bounding Box [28] </i>

### Combination with Deep Learning Computer Vision - Fusion Sensor

LiDAR can be merged with Computer Vision to create fusion sensor, turning the sensor into a reliable method that complements the strength and weaknesses of each other.

## Research into usage of standardized ArUco fiducial markers for ADAS

The author will research about the use of standardized size ArUco markers for adding another layer of depth perception for various tasks.

<img src="./imgs/ArUco.JPG" width=850 height=620 /> <br>
<i> Figure 36: Author proposed method using ArUco fiducial markers for naive depth & perception estimation [i] </i>

## Ongoing Project

Next step in the research and development work will be to improve this model and to use Mask R-CNN to include image segmentation.

<b> Task 1: Mask R-CNN by expanding on the current Functional API model architecture. -- Ongoing </b> <br>
Task 2: Panoptic Segmentation with EfficientPS <br>
Task 3: Single Shot Detectors / YOLO <br>
Task 4: LiDAR Point Cloud Segmentation <br>
Task 5: Lane Detection - Rule Based & Deep Learning methods <br>
Task 6: Stereo Camera CV <br>
Task 7: Rain Removal based on research paper 'A Model-driven Deep Neural Network for Single Image Rain Removal' [j] <br>
Task 8: Pedestrian Pose Estimation - Edge cases <br>
Task 9: ArUco fiducial markers <br>

Improvement Research Tasks: <br>
1. Mask Segmentation during object occlusion <br>
2. Robust Stereo CV depth perception <br>
3. LiDAR with CV - Fusion Sensor Perception <br>


## Sources: 

[1] https://traveltips.usatoday.com/air-travel-safer-car-travel-1581.html#:~:text=In%20absolute%20numbers%2C%20driving%20is,air%20travel%20to%20be%20safer.

[2] https://towardsdatascience.com/r-cnn-fast-r-cnn-faster-r-cnn-yolo-object-detection-algorithms-36d53571365e

[3] https://towardsdatascience.com/region-of-interest-pooling-f7c637f409af

[4] https://towardsdatascience.com/with-keras-functional-api-your-imagination-is-the-limit-4f4fae58d90b

[5] https://towardsdatascience.com/review-mobilenetv2-light-weight-model-image-classification-8febb490e61c

[6] https://medium.com/analytics-vidhya/train-a-custom-yolov4-object-detector-using-google-colab-61a659d4868

[7] https://traveltips.usatoday.com/air-travel-safer-car-travel-1581.html

[8] https://medium.com/analytics-vidhya/iou-intersection-over-union-705a39e7acef

[9] https://medium.datadriveninvestor.com/2-layers-to-greatly-improve-keras-cnn-1d4d1c3e8ea5

[10] https://towardsdatascience.com/the-vanishing-exploding-gradient-problem-in-deep-neural-networks-191358470c11?gi=27e999e4010f

[11] https://developers.google.com/machine-learning/crash-course/multi-class-neural-networks/softmax

[12] https://towardsdatascience.com/understand-your-algorithm-with-grad-cam-d3b62fce353#:~:text=Gradient%2Dweighted%20Class%20Activation%20Mapping,regions%20in%20the%20image%20for

[13] https://arxiv.org/pdf/1610.02391.pdf

[14] https://towardsdatascience.com/fast-r-cnn-for-object-detection-a-technical-summary-a0ff94faa022

[15] https://github.com/tzutalin/labelImg

[16] https://towardsdatascience.com/what-is-the-difference-between-object-detection-and-image-segmentation-ee746a935cc1

[17] https://www.researchgate.net/figure/Object-tracking-based-on-image-segmentation-and-similar-object-feature-matching_fig1_4222458

[18] https://m.futurecar.com/4632/Computer-Vision-Developer-StradVision-to-Showcase-its-Most-Advanced-Perception-Camera-for-Autonomous-Driving-&amp;-ADAS-at-Auto-Tech-2021

[19] https://github.com/matterport/Mask_RCNN

[20] https://www.makesense.ai/

[21] https://towardsdatascience.com/review-resnet-winner-of-ilsvrc-2015-image-classification-localization-detection-e39402bfa5d8

[22] https://towardsdatascience.com/the-vanishing-gradient-problem-69bf08b15484

[23] https://ai.googleblog.com/2019/05/efficientnet-improving-accuracy-and.html

[24] https://towardsdatascience.com/google-releases-efficientnetv2-a-smaller-faster-and-better-efficientnet-673a77bdd43c

[25] https://caradas.com/adas-have-changed-the-auto-industry-forever-heres-how/

[26] https://www.mathworks.com/help/deeplearning/ug/lidar-semantic-segmentation-using-squeezesegv2.html

[27] https://cbmjournal.biomedcentral.com/articles/10.1186/s13021-018-0098-0/figures/2

[28] https://medium.datadriveninvestor.com/lidar-3d-perception-and-object-detection-311419886bd2

[29] https://jonathan-hui.medium.com/image-segmentation-with-mask-r-cnn-ebe6d793272#:~:text=Mask%20R%2DCNN%20uses%20ROI,values%20within%20the%20cell%20better.

---

[a] Research papers and materials: With great thanks to the following authors for sharing their research papers and materials on the topic of object classification and localization.

[b] Object Detection and Localization with Deep Networks, Avi Kak and Charles Bouman, Purdue University

[c] Universal Bounding Box Regression and Its Applications, Seungkwan Lee, Suha Kwak and Minsu Cho, Dept. of Computer Science and Engineering, POSTECH Korea

[d] Intelligent Vision Systems & Embedded Deep Learning Technology for ADAS, Jiun-In Guo, National Yang Ming Chiao Tung University

[e] Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization, Ramprasaath R. Selvaraju · Michael Cogswell · Abhishek Das · Ramakrishna
Vedantam · Devi Parikh · Dhruv Batra

[f] Stereo RCNN based 3D Object Detection for Autonomous Driving, https://github.com/srinu6/Stereo-3D-Object-Detection-for-Autonomous-Driving

[g] An automatic nuclei segmentation method based on deep convolutional neural networks for histopathology images - Scientific Figure on ResearchGate. Available from: https://www.researchgate.net/figure/The-overall-network-architecture-of-Mask-R-CNN_fig1_336615317 [accessed 13 Apr, 2022]

[h] Deep Residual Learning for Image Recognition, Authors: Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun

[i] Autonomous UAV with vision based on-board decision making for remote sensing and precision agriculture, Bilal H. Alsalam, Kye Morton, Duncan Andrew Campbell, Luis Felipe Gonzalez

[j] A Model-driven Deep Neural Network for Single Image Rain Removal, Hong Wang, Qi Xie, Qian Zhao, Deyu Meng

[k] Unsupervised Learning of Anomaly Detection from Contaminated Image Data using Simultaneous Encoder Training, Amanda Berg, Jorgen Ahlberg, Michael Felsberg, Termisk, Systemteknik AB, Diskettgatan 11 B, 583 35 Linkoping, Sweden, Computer Vision Laboratory, Dept. EE

[m] Image Segmentation and Pattern Matching Based FPGA/ASIC Implementation Architecture of Real-Time Object Tracking, K. Yamaoka, T. Morimoto, H. Adachi, T. Koide, and H. J. Mattausch, Research Center for Nanodevices and Systems, Hiroshima University

[n] An Improved DBSCAN Method for LiDAR Data Segmentation with Automatic Eps Estimation, Chunxiao Wang, Min Ji, Jian Wang, Wei Wen, Ting Li, and Yong Sun

[p] Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks, Shaoqing Ren, Kaiming He, Ross Girshick, and Jian Sun

[q] Mask R-CNN, Kaiming He Georgia Gkioxari Piotr Dollar Ross Girshick, Facebook AI Research (FAIR)

---

Dataset: 
There is no intention of copyright infringement, all images are for education & research purposes with no intention for commercial deployment.

1. https://github.com/eugeneyan84
2. Google Street View
3. Taken with mobile phone

---

Saved Models and Weights:

https://drive.google.com/file/d/1kokwbQd20Y_jmHTJo3x8ysc2_NUyn-ur/view?usp=sharing

Images Datasets:

https://drive.google.com/drive/folders/1upxY-xXbUB1Yd1afpCJG--8tgndd_2DN?usp=sharing

