# Automatic Number Plate Recognition

## Software Requirements
1) opencv-python 4.5.3.56 or above
2) numpy 1.20.0 or above
3) python 3.9 or above
4) numpy 1.20.0 or above
5) pytorch 1.10.0 or above

## **Goal of Project**
The goal of this project was to create a model that can accurately detect number plates on cars and bikes. This model can be used along with the video data generated from CCTV cameras that are installed on highways and roadways to detect number plate of vechicles that commit traffic violations.

## **Project Information**
Object Detection is carried out in this project to detect Number Plates on vehicles. YOLOv5s model was used along with transfer learning for training and testing. Model accurately generates bounding boxes around Number Plates 

More Info on YOLOv5: https://github.com/ultralytics/yolov5

## **Implementation**

Object Detection pipeline has 3 parts Training, Validation and Testing

### Training:

YOLOv5s was trained on Google Colab with following hyperparameters:

1) Input Image Size: 640
2) Batch Size: 16
3) Epochs: 300 (converged in 198 epochs)
4) Pretrained Weights: yolov5s.pt

The training dataset consisted of 400 images along with their class and bounding box details mentioned in a txt file in yolo format. The training was set to run for 300 epochs but the model converged in 198 epochs and the training was stopped.

### Validation:

The validation dataset consisted of 100 images along with their class and bounding box details in a txt file in yolo format for validation purpose. In validation phase the model reached Mean Average Precision (mAP) of 0.91

Following are some of the image results generated in validation phase

![alt text](https://github.com/wasdac9/automatic-number-plate-recognition/blob/main/val_batch0_pred.jpg?raw=true)

### Testing Phase
The model was tested on various images and videos and the model generated accurate class and bounding box predictions.

#### **Image Inference**
Few of the images used for testing are included in the "test_images" folder. At test time the model generated 6-7 FPS on a batch of 30 images with a Meam Time of 0.15ms

#### **Video Inference**
A video that was used for testing is also included in the name "anpr_video.mp4". At inference this video was passed through the trained model and the output generated is saved as "output.mp4". At test time the model generated 8-9 FPS on i5 CPU and around 32 FPS on CUDA device.


