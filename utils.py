import argparse
from PIL import Image
import glob, os
import numpy as np

def get_input_args(option):
    parser = argparse.ArgumentParser()

    if option == 'train':
        parser.add_argument('data_dir', type=str, help='directory of the image folder')
        parser.add_argument('--save_dir', default='checkpoint.tar', type=str, help='directory to save the checkpoint')
        parser.add_argument('--arch', default='vgg16', type=str, help='select between vgg16 and resnet50')
        parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate of the neural network')
        parser.add_argument('--hidden_units', default=[4096,1024], nargs='+', type=int, help='hidden units of the neural network')
        parser.add_argument('--epoch', default=5, type=int, help='epochs of training')
        parser.add_argument('--gpu', default=True, action='store_true', help='use gpu for training')

    elif option == 'predict':
        parser.add_argument('image_dir', type=str, help='directory of the checking image')
        parser.add_argument('checkpoint', default='checkpoint.tar', type=str, help='directory of the checkpoint')
        parser.add_argument('--top_k', type=int, default=3, help='number of top most likely classes')
        parser.add_argument('--category_names', type=str, default='cat_to_name.json', help='mapping of categories to real names')
        parser.add_argument('--gpu', default=True, action='store_true', help='use gpu for inference')
        
    return parser.parse_args()

def process_image(image_path):
    image = Image.open(image_path)
    size = 256, 256
    image.thumbnail(size)
    image = image.resize((224, 224))
    np_image = np.array(image) 
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image/255 - mean) / std
    np_image = np_image.transpose()
    return np_image

