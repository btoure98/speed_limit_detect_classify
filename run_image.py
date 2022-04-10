from turtle import speed
import cv2
import torch
import config as cfg
import argparse

from image_processing import hough_detector

decoding = {0: 30,
            1: 50,
            2: 70,
            3: 90,
            4: 110}

def detect_and_classify(image):
    circles = hough_detector(image)
    #load model
    model = torch.load(cfg.model_path)
    model.eval()
    #detected speeds list
    speeds = {}
    detected = False
    #inference
    for circle in circles:
        sign = torch.from_numpy(circle)
        sign = (1.0/255)*sign.permute(2, 0, 1).unsqueeze(0)
        prediction = model(sign)
        if prediction.max() > 0.8:
            detected = True
            predicted_speed = decoding[prediction[0].argmax().item()]
            speeds[(predicted_speed)] = prediction.max().item()
    return speeds, detected

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Detect and classify SLS')
    parser.add_argument(
        'image_path', help='path to image')
    args = parser.parse_args()
    #read image
    image = cv2.imread(args.image_path)
    speeds, detected = detect_and_classify(image)
    if detected:
        print(f"possible speeds in image with confidence: {speeds}")
    else:
        print("No SlS detected")