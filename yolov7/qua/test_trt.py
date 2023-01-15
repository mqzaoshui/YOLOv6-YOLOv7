#xyh 
from utils.utils import preproc, vis
from utils.utils import BaseEngine
import numpy as np
import cv2
import time
import os
import argparse
from utils.batch import Batch
import sys

def t2t(line,file):
    with open(file,'a') as f:
        f.write(line + '\n')


class Predictor(BaseEngine):
    def __init__(self, engine_path , imgsz=(640,640)):
        super(Predictor, self).__init__(engine_path)
        self.imgsz = imgsz # your model infer image size
        self.n_classes = 80  # your model classes

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--engine", help="TRT engine Path")
    parser.add_argument('-s', '--source',type=str, default='inference/images', help='images source')
    parser.add_argument("-i", "--image", help="image path")
    parser.add_argument("-o", "--output", help="image output path")
    parser.add_argument("-v", "--video",  help="video path or camera index ")
    parser.add_argument("--end2end", default=False, action="store_true",
                        help="use end2end engine")
    parser.add_argument("--conf",type=float,default=0.4,help='object confidence threshold')
    args = parser.parse_args()
    print(args)

    pred = Predictor(engine_path=args.engine)
    pred.get_fps()
    input_dir = args.source
    channel = Batch(input_dir)
    image_dir = channel.get_image()

    if input_dir:
      for i in image_dir:
        img_path = i
        video = args.video
        if img_path:
          origin_img, t1, t2, t3= pred.inference(img_path, conf=args.conf, end2end=args.end2end)
          # Infertime, whole Infer, postprocessing time
          t = [round(1E3*t1,2),round(1E3*t2,2),round(1E3*t3,2)]

          out_basename = 'infer' + os.path.basename(i)
          out = os.path.join(args.output,out_basename)
          cv2.imwrite("%s" %out , origin_img)
          #print(f'{1E3 * (t):.1f}ms,Inference')
          txt = os.path.join(args.output,'time.txt')
          t2t(str(t),txt)
        if video:
          pred.detect_video(video, conf=args.conf, end2end=args.end2end) # set 0 use a webcam
    
