import torch
import cv2
import numpy as np
from pathlib import Path
# Model
model_path = Path("best.pt") #custom model path
img_path = Path("test_images/car_384.jpg") #input video path
cpu_or_cuda = "cpu"  #choose device; "cpu" or "cuda"(if cuda is available)
device = torch.device(cpu_or_cuda)

model = torch.hub.load('ultralytics/yolov5', 'custom', path= model_path, force_reload=True)
model = model.to(device)
image = cv2.imread(img_path)

# Inference
output = model(image)

# Results
result = np.array(output.pandas().xyxy[0])
for i in result:
    p1 = (int(i[0]),int(i[1]))
    p2 = (int(i[2]),int(i[3]))
    text_origin = (int(i[0]),int(i[1])-5)
    text_font = cv2.FONT_HERSHEY_PLAIN
    color= (0,0,255)
    text_font_scale = 1.25
    #print(p1,p2)
    image = cv2.rectangle(image,p1,p2,color=color,thickness=2) #drawing bounding boxes
    image = cv2.putText(image,text=f"{i[-1]}",org=text_origin,
                        fontFace=text_font,fontScale=text_font_scale, 
                        color=color,thickness=2)  #class and confidence text

cv2.imshow("image",image)
cv2.waitKey(0)
cv2.destroyAllWindows()