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

Object Detection pipeline cinsists of 3 parts:
**Training, Validation and Testing**

### Training:

YOLOv5s was trained on Google Colab with following hyperparameters:

1) Input Image Size: 640
2) Batch Size: 16
3) Epochs: 300 (converged in 198 epochs)
4) Pretrained Weights: yolov5s.pt

The training dataset consisted of 400 images along with their class and bounding box details mentioned in a txt file in yolo format. The training was set to run for 300 epochs but the model converged in 198 epochs and the training was stopped.

### Validation:

The validation dataset consisted of 100 images along with their class and bounding box details in a txt file in yolo format for validation purpose. In validation phase the model reached Mean Average Precision (mAP) of 0.91

Following are some of the image results generated in validation phase:

![alt text](https://github.com/wasdac9/automatic-number-plate-recognition/blob/main/val_pred.jpg?raw=true)

At the end of training and validation epochs, a weights file ("best.pt") is generated which consists of all the learned parameters of the model. 

Refer to "ANPR_object_detection.ipynb" for more info about the training and validation process.

### Testing Phase
The model was tested on various images and videos and the model generated accurate class and bounding box predictions. The weights file called "best.pt" that was generate in the training phase was used for inference in testing phase. Testing was carried out in PyTorch, and OpenCV was used when working with images and videos. OpenCV helped in loading, saving, drawing bounding boxes, and displaying text regarding class name and class confidence values.

#### **Image Inference**
Few of the images used for testing are included in the "test_images" folder. At test time the model generated 6-7 FPS on a batch of 30 images with a Mean Time of 0.15ms. Python file named "image_anpr.py" is used to generate image inferences.

#### **Video Inference**
A video that was used for testing is also included in the name "anpr_video.mp4". At inference this video was passed through the trained model and the output generated is saved as "output.mp4". At test time the model generated 8-9 FPS on i5 CPU and around 32 FPS on CUDA device. Python file named "video_anpr.py" is used to generate video inferences.

## **Code**

Set the following paths and variables before running both the python files for image and video inference.

Changes to be made in "image_anpr.py" for image inference
```
model_path = Path("best.pt") #custom model path
img_path = Path("test_images/car_384.jpg") #input image path
cpu_or_cuda = "cpu"  #choose device; "cpu" or "cuda"(if cuda is available)
device = torch.device(cpu_or_cuda)
model = torch.hub.load('ultralytics/yolov5', 'custom', path= model_path, force_reload=True)
model = model.to(device) #loading model to cpu or cuda
```

Changes to be made in "video_anpr.py" for video inference
```
model_path = r"best.pt"  #custom model path
video_path = r"anpr_video.mp4"  #input video path
cpu_or_cuda = "cpu"  #choose device; "cpu" or "cuda"(if cuda is available)
device = torch.device(cpu_or_cuda)
model = torch.hub.load('ultralytics/yolov5', 'custom', path= model_path, force_reload=True)
model = model.to(device) #loading model to cpu or cuda
```

## **Summary**
An accurate object detection model was created to carry out Automatic Number Plate Recognition using YOLOv5 and transfer learning along with Pytorch. The accuracy of bounding boxes and the frame rate was found to be good. 

## **Limitations**
1) Even though the accuracy and frame rate was good, the model sometimes wrongly detected various signs on road as number plate. A bigger dataset is required to improve upon the current model accuracy.
2) The model in its current state works very well on CUDA devices getting around 32 FPS but does perform well on CPU devices. Hence GPU hardware is required for smooth frame rate outputs.

## **Future Work**
1) A bigger dataset can be used to train the model for more number of epochs to reduce the false positive predictions.
2) This detection model can be uploaded on edge devices connected to CCTV cameras to carry out Number Plate Recognition live on the road.
3) CCTV video footage can be used to read number plate of vehicles that commit traffic violations.
4) The bounding box around the license plate can be cropped and Optical Character Recognition can be used to actually read the number plate.
