from ultralytics import YOLO
import onnxruntime
import time
import cv2
import torch
#model = YOLO('train3.onnx')
model = YOLO('train3new.pt')
results =  model (source=1, show=True, conf=0.80,stream = True )

for r in results:
    boxes = r.boxes
    mask = r.mask
    probs = r.probs
mask = r.masks[1]  # Accede a la primera máscara de detección en el objeto Results

8