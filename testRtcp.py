import cv2
import psutil
import os
import grpc
from concurrent import futures
import grcp.proto3.model_service_proto_pb2 as model_ppb2
import grcp.proto3.model_service_proto_pb2_grpc as model_grcp

import numpy as np

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)  # En megabytes

# RTSP URL
rtsp_url = 'rtsp://admin:admin@172.172.3.21:8554/CH001.sdp'

# Configura la conexión gRPC
with grpc.insecure_channel("localhost:50052") as channel:
    stub = model_grcp.TrtYoloServiceStub(channel)

    # Open RTSP stream
    cap = cv2.VideoCapture(rtsp_url)

    # Verifica si la transmisión se abre correctamente
    if not cap.isOpened():
        print("Error: No se pudo abrir la transmisión RTSP.")
        exit()

    # Lee y muestra frames
    while True:
        memory_usage = get_memory_usage()
        print(f"Uso de memoria: {memory_usage:.2f} MB")

        # Lee un frame de la transmisión RTSP
        ret, frame = cap.read()

        # Verifica si el frame se lee correctamente
        if not ret:
            print("Error: No se pudo leer el frame.")
            break
        
        _, image_bytes = cv2.imencode('.jpg', frame)

        # Asegúrate de que el frame sea válido antes de intentar procesarlo
        if frame is None:
            print("Error: El frame no es válido.")
            break
        image_bytes = image_bytes.tobytes()
        # Prepara la solicitud gRPC
        
        stub.UploadImage(model_ppb2.ImageUploadRequest(image=image_bytes))       
        #print(future_response)
        # Envía la solicitud al servidor gRPC
        

        # Muestra el frame
        # cv2.imshow('RTSP Stream', frame)

        # Sale del bucle si se presiona la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Libera los recursos después de procesar todos los frames
cap.release()
cv2.destroyAllWindows()
