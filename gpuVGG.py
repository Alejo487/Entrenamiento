import torch
import cv2
import torchvision.transforms as transforms
from torchvision import models
import numpy as np
import time

### Se usa la GPU 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

#### Modelo VGG16
model = models.vgg16(weights=None)

model.classifier = torch.nn.Sequential(
    torch.nn.Linear(25088, 4096),
    torch.nn.ReLU(True),
    torch.nn.Dropout(),
    torch.nn.Linear(4096, 4096),
    torch.nn.ReLU(True),
    torch.nn.Dropout(),
    torch.nn.Sequential(
        torch.nn.Linear(4096, 256),
        torch.nn.ReLU(True),
        torch.nn.Dropout(),
        torch.nn.Linear(256, 2)
    )
)

#### Aqui se carga el modelo entrenado el VGG
model.load_state_dict(torch.load("vgg16-transfer-4.pt", map_location=device))
model.to(device)
model.eval()

### Con la función "Transform" se puede darle transformacion a las imágenes 
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


cap = cv2.VideoCapture(0)


labels = ['vanessa', 'alejandro']  # Cambia estos nombres a los de tus etiquetas

###se usa haarcascade para usar la funcion de reconocimiento frnotal face
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


prev_time = 0
fps = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break


    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

   
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

 
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
       
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

       
        face = frame[y:y+h, x:x+w]

       
        img = transform(face)
        img = img.unsqueeze(0).to(device)  # Mover la imagen a la GPU

       
        with torch.no_grad():
            outputs = model(img)
            _, predicted = torch.max(outputs, 1)
            label = labels[predicted.item()]

    
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)

   
    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('Camera', frame)

   
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
