import utils
import matplotlib.pyplot as plt
from workspace_utils import keep_awake
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import json

device = None
args = utils.get_input_args('predict')
if (args.gpu):
    device = 'cuda'
else:
    device = 'cpu'
    
class_to_label = None
with open(args.category_names, 'r') as f:
    class_to_label = json.load(f)

def load_checkpoint(file_path):
    checkpoint = torch.load(file_path)
    model = None
    optimizer = None
    if checkpoint['model'] == 'vgg16':
        model = models.vgg16(pretrained=True)
        for params in model.parameters():
            params.requires_grad = False
        model.classifier = checkpoint['classifier']
        optimizer = optim.Adam(model.classifier.parameters(), lr=checkpoint['learn_rate'])
    if checkpoint['model'] == 'resnet50':
        model = models.resnet50(pretrained=True)
        for params in model.parameters():
            params.requires_grad = False
        model.fc = checkpoint['classifier']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer = optim.Adam(model.fc.parameters(), lr=checkpoint['learn_rate'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    criterion = nn.NLLLoss()
    epoch = checkpoint['epoch']
    model.class_to_idx = checkpoint['class_to_idx']
    model.to(device)
    
    return model, criterion, optimizer, epoch
    
model, criterion, optimizer, epoch = load_checkpoint(args.checkpoint)

idx_to_class = {v: k for k, v in model.class_to_idx.items()}

model.eval()
image = utils.process_image(args.image_dir)
image_torch = torch.from_numpy(image).unsqueeze_(0)
image = image_torch.to(device)
with torch.no_grad():
    output = model.forward(image.float())
ps = torch.exp(output)
top = ps.topk(args.top_k)
probs = top[0].cpu().numpy()[0]
classes = []
for idx in top[1].cpu().numpy()[0]:
    classes.append(idx_to_class[idx])
for i in range(len(probs)):
    print(class_to_label[classes[i]], ': {:.2f}%'.format(probs[i]*100))
