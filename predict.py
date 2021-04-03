from flask import Flask, render_template, request
# from werkzeug import secure_filename
import torch
from PIL import Image
import torchvision
import torchvision.transforms as transforms
# import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
# import torch.optim as optim
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
# checkpoint = {'model': model,
#           'state_dict': model.state_dict(),
#           'optimizer' : optimizer.state_dict()}

# torch.save(checkpoint, 'checkpoint.pth')

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 2 * 2, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 10)
        self.dropout1 = nn.Dropout(p=0.2, inplace=False)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.dropout1(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout1(x)
        x = self.pool(F.relu(self.conv3(x)))
        x = self.dropout1(x)
        x = x.view(-1, 128 * 2 * 2)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x) #output layer
        
        return x


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath, map_location='cpu')
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    for parameter in model.parameters():
        parameter.requires_grad = False

    model.eval()
    return model

model = load_checkpoint('checkpoint.pth')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model.to(device)

def do_predcition(filepath):
    pil_image = Image.open(filepath)
    img_loader = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor()])
    
      
    ts_image = img_loader(pil_image).float()
    ts_image.unsqueeze_(0)
    outputs = model(ts_image.to(device))
    _, predicted = torch.max(outputs.data, 1)

    return classes[predicted[0]]

@app.route('/')
def upload_file_page():
   return render_template('index.html')
	
@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
   image_path = ''
   if request.method == 'POST':
      f = request.files['file']
      f.save(secure_filename(f.filename))
      image_path = "./" + secure_filename(f.filename)
      
      category = do_predcition(image_path)

      os.remove(image_path)

      print(category)
      return category
		
if __name__ == '__main__':
   app.run(debug = True)