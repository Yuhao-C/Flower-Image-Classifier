import utils
import matplotlib.pyplot as plt
from workspace_utils import keep_awake
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import json
from collections import OrderedDict

device = None
args = utils.get_input_args('train')
if (args.gpu):
    device = 'cuda'
else:
    device = 'cpu'
    
data_dir = args.data_dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                     [0.229, 0.224, 0.225])])

valid_transforms = transforms.Compose([transforms.Resize(256),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                     [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                    [0.229, 0.224, 0.225])])

train_datasets = datasets.ImageFolder(train_dir, transform=train_transforms)
valid_datasets = datasets.ImageFolder(valid_dir, transform=valid_transforms)
test_datasets = datasets.ImageFolder(test_dir, transform=test_transforms)

trainloaders = torch.utils.data.DataLoader(train_datasets, batch_size=32, shuffle=True)
validloaders = torch.utils.data.DataLoader(valid_datasets, batch_size=32)
testloaders = torch.utils.data.DataLoader(test_datasets, batch_size=32)

def build_vgg16():
    od = OrderedDict()
    
    model = models.vgg16(pretrained=True)
    for params in model.parameters():
        params.requires_grad = False
        
    count = 1
    last_hidden_unit = None
    for hidden_unit in args.hidden_units:
        if count == 1:
            od['fc'+str(count)] = nn.Linear(25088, hidden_unit)   
        else:
            od['fc'+str(count)] = nn.Linear(last_hidden_unit, hidden_unit)
        last_hidden_unit = hidden_unit
        od['relu'+str(count)] = nn.ReLU(inplace=True)
        od['dropout'+str(count)] = nn.Dropout()
        count += 1
    od['fc'+str(count)] = nn.Linear(last_hidden_unit, 102)
    od['softmax'] = nn.LogSoftmax(dim=1)
    classifier = nn.Sequential(od)
    model.classifier = classifier
    return model

def build_resnet50():
    od = OrderedDict()
    
    model = models.resnet50(pretrained=True)
    for params in model.parameters():
        params.requires_grad = False
        
    count = 1
    last_hidden_unit = None
    for hidden_unit in args.hidden_units:
        if count == 1:
            od['fc'+str(count)] = nn.Linear(2048, hidden_unit)   
        else:
            od['fc'+str(count)] = nn.Linear(last_hidden_unit, hidden_unit)
        last_hidden_unit = hidden_unit
        od['relu'+str(count)] = nn.ReLU(inplace=True)
        od['dropout'+str(count)] = nn.Dropout()
        count += 1
    od['fc'+str(count)] = nn.Linear(last_hidden_unit, 102)
    od['softmax'] = nn.LogSoftmax(dim=1)
    classifier = nn.Sequential(od)
    model.fc = classifier
    return model

def validate(validloaders, device, model, criterion):
    valid_loss = 0
    valid_accuracy = 0
    num = 0
    for images, labels in validloaders:
        
        images, labels = images.to(device), labels.to(device)
        
        outputs = model.forward(images)
        valid_loss += criterion(outputs, labels).item()
        
        ps = torch.exp(outputs)
        equality = (ps.max(dim=1)[1] == labels.data)
        valid_accuracy += equality.type(torch.FloatTensor).mean()
        
    return valid_loss/len(validloaders), valid_accuracy/len(validloaders)

def train():
    epoch = args.epoch
    print_every = 40
    step = 0
    running_loss = 0
    for e in keep_awake(range(epoch)):
        model.train()    
        for images, labels in trainloaders:
            step += 1        
            images, labels = images.to(device), labels.to(device)      
            optimizer.zero_grad()        
            outputs = model.forward(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()        
            running_loss += loss.item()      
            if step % print_every == 0:
                model.eval()            
                with torch.no_grad():
                    valid_loss, valid_accuracy = validate(validloaders, device, model, criterion)            
                print("Epoch: {}/{}.. ".format(e+1, epoch),
                    "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                    "Validation Loss: {:.3f}.. ".format(valid_loss),
                    "Validation Accuracy: {:.3f}".format(valid_accuracy))
                running_loss = 0            
                model.train()
            
def test():
    model.eval()
    with torch.no_grad():
        test_accuracy = 0
        num = 0
        for images, labels in testloaders:
            num += 1
        
            images, labels = images.to(device), labels.to(device)
            outputs = model.forward(images)

            ps = torch.exp(outputs)
        
            equality = (ps.max(dim=1)[1] == labels.data)
            test_accuracy += equality.type(torch.FloatTensor).mean()  
    print("Test Accuracy: {:.2f}%".format(test_accuracy/num*100))
    
def save(save_dir):
    model.class_to_idx = train_datasets.class_to_idx
    if (args.arch == 'vgg16'):
        checkpoint = {
        'model': 'vgg16',
        'epoch': args.epoch,
        'optimizer_state_dict': optimizer.state_dict(),
        'learn_rate': args.learning_rate,
        'class_to_idx': model.class_to_idx,
        'classifier': model.classifier}
    elif (args.arch == 'resnet50'):
        checkpoint = {
        'model': 'resnet50',
        'epoch': args.epoch,
        'optimizer_state_dict': optimizer.state_dict(),
        'learn_rate': args.learning_rate,
        'class_to_idx': model.class_to_idx,
        'classifier': model.fc,
        'model_state_dict': model.state_dict()}
    torch.save(checkpoint, save_dir)
    
    
model = None
optimizer = None

if (args.arch == 'vgg16'):
    model = build_vgg16()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)
elif (args.arch == 'resnet50'):
    model = build_resnet50()
    optimizer = optim.Adam(model.fc.parameters(), lr=args.learning_rate)
else:
    print('Please select between vgg16 and resnet50 for architechture')
    sys.exit()
    
criterion = nn.NLLLoss()
model.to(device)
train()
test()
save(args.save_dir)