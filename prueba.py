# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import cv2
import numpy as np

# Cargar el modelo YOLO y configuración
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getUnconnectedOutLayersNames()

# Cargar el video
cap = cv2.VideoCapture("multi.mp4")


while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width, _ = frame.shape

    # Preprocesar la imagen
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(layer_names)

    # Contar carros (o el objeto que estás buscando)
    count_cars = 0
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id == 2:  # 2 es la clase de automóviles en el modelo COCO
                count_cars += 1

    # Mostrar el resultado
    cv2.putText(frame, f'Carros: {count_cars}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Contador de Carros", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Presiona 'Esc' para salir
        break

cap.release()
cv2.destroyAllWindows()
