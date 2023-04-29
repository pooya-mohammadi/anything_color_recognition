from argparse import ArgumentParser
from time import time

import numpy as np
from PIL import Image, ImageDraw
from deep_utils import ColorRecognitionCNNTorchPrediction, CVUtils

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--model_path", default="output/exp_1/best.ckpt")
    parser.add_argument("--device", default="cpu", help="cuda or cpu")
    parser.add_argument("--img_path",
                        default=r"C:\Users\pooya\projects\edge-device-knit\knitvision_imagedataset\object-detection-balanced\dataset\train\images/385_HD.jpg")
    args = parser.parse_args()
    model = ColorRecognitionCNNTorchPrediction(args.model_path, device=args.device)
    img = Image.open(args.img_path)
    tic = time()
    prediction = model.detect(args.img_path)
    toc = time()
    draw = ImageDraw.Draw(img)
    draw.text((20, 20), prediction, (0, 255, 0))
    print(f"Vehicle Color is: {prediction}, inference time: {toc - tic}")
    CVUtils.show_destroy_cv2(np.array(img)[..., ::-1])
