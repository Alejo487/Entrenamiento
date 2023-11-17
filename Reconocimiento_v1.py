from ultralytics import YOLO
import time
import cv2
import torch
model = YOLO('nuevoEntrenamiento.pt')

results = model (source=1, show=True, conf=0.76, save=True,)
#boxes=True
