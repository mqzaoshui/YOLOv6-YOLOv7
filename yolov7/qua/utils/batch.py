import numpy as np
import re
import os


class Batch:
    def __init__ (self, input_dir):
        input = os.path.realpath(input_dir)
        
        self.images = []
        extensions = [".jpg", ".jpeg", ".png", ".bmp"]

        def is_image(path):
                return os.path.isfile(path) and os.path.splitext(path)[1].lower() in extensions
    
        if os.path.isdir(input):
                self.images = [os.path.join(input, f) for f in os.listdir(input) if is_image(os.path.join(input, f))]
                self.images.sort()
        elif os.path.isfile(input):
            if is_image(input):
                self.images.append(input)
        self.num_images = len(self.images)
        if self.num_images < 1:
            print("No valid {} images found in {}".format("/".join(extensions), input))


    def get_image(self):

        return  self.images
