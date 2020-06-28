import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import nn
from torch import optim
import torch.utils.data
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import torchvision.models as models
from collections import OrderedDict
import json
import PIL
from PIL import Image
import seaborn as sns

parser = argparse.ArgumentParser (description = "Parser of prediction script")

parser.add_argument ('image_dir', help = 'Provide path to image. Mandatory argument', type = str)
parser.add_argument ('load_dir', help = 'Provide path to checkpoint. Mandatory argument', type = str)
parser.add_argument ('--top_k', help = 'Top K most likely classes. Optional', type = int)
parser.add_argument ('--category_names', help = 'Mapping of categories to real names. JSON file name to be provided. Optional', type = str)
parser.add_argument ('--GPU', help = "Option to use GPU. Optional", type = str)

def load_check (filepath):
    checkpoint = torch.load (filepath) 
    if checkpoint ['arch'] == 'alexnet':
        model = models.alexnet (pretrained = True)
    else: 
        model = models.vgg13 (pretrained = True)
    model.classifier = checkpoint ['classifier']
    model.load_state_dict (checkpoint ['state_dict'])
    model.class_to_idx = checkpoint ['mapping']

    for param in model.parameters():
        param.requires_grad = False

    return model


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    im = Image.open (image) 
    w, h = im.size 
    
    if w > h: 
        h = 256
        im.thumbnail ((50000, h))
    else: 
        w = 256
        im.thumbnail ((w,50000))
        
    
    w, h = im.size 
    reduce = 224
    l = (w - reduce)/2 
    t = (h - reduce)/2
    r = l + 224 
    b = t + 224
    im = im.crop ((l, t, r, b))
    
    np_image = np.array (im)/255 
    np_image -= np.array ([0.485, 0.456, 0.406]) 
    np_image /= np.array ([0.229, 0.224, 0.225])
    
    np_image= np_image.transpose ((2,0,1))
    np_image.transpose (1,2,0)
    return np_image

def predict(image_path, model, topkl=5, device):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    image = process_image (image_path) 

    if device == 'cuda':
        im = torch.from_numpy (image).type (torch.cuda.FloatTensor)
    else:
        im = torch.from_numpy (image).type (torch.FloatTensor)

    im = im.unsqueeze (dim = 0) 

    model.to (device)
    im.to (device)

    with torch.no_grad ():
        output = model.forward (im)
    output_prob = torch.exp (output) 

    probs, indeces = output_prob.topk (topkl)
    probs = probs.cpu ()
    indeces = indeces.cpu ()
    probs = probs.numpy ()
    indeces = indeces.numpy ()

    probs = probs.tolist () [0]
    indeces = indeces.tolist () [0]

    mapping = {val: key for key, val in
                model.class_to_idx.items()
                }

    classes = [mapping [item] for item in indeces]
    classes = np.array (classes) 

    return probs, classes

args = parser.parse_args ()
file_path = args.image_dir

if args.GPU == 'GPU':
    device = 'cuda'
else:
    device = 'cpu'
if args.category_names:
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)
else:
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
        pass

model = load_check (args.load_dir)

if args.top_k:
    nm_cl = args.top_k
else:
    nm_cl = 1

probs, classes = predict (file_path, model, nm_cl, device)

class_names = [cat_to_name [item] for item in classes]

for l in range (nm_cl):
     print("Number: {}/{}.. ".format(l+1, nm_cl),
            "Class name: {}.. ".format(class_names [l]),
            "Probability: {:.3f}..% ".format(probs [l]*100),
            )
