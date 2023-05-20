import torch
import cv2

model = torch.hub.load('yolov5', 'custom', path='./models/best.pt', 
                       source='local', force_reload=True)
model.eval()
model.conf = 0.8

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if ret is None:
        print('Exit...')
        break

    results = model(frame)
    results.print()

    r_img = results.render()
    img_result = r_img[0]
    cv2.imshow('Result', img_result)
    
    key = cv2.waitKey(1)
    if key == ord('q'):
        break