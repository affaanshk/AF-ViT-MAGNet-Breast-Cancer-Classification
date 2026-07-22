# examples/predict_example.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from inference import predict_single_image

if __name__ == "__main__":
    print("Running AF-ViT-MAGNet Inference Example...")
    # Example execution call
    # results = predict_single_image(model_path="weights/best_vit.pth", image_path="path/to/clean_sample.png")
    # print("Prediction Output:", results)