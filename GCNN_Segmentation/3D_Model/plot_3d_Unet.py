import glob
import os
import time

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

from PIL import Image
from scipy.spatial.distance import cdist
from torchvision import datasets, transforms
from tqdm import tqdm
import torchvision.transforms.functional as TF

from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from skimage import measure
import matplotlib.pyplot as plt

data_file = 'C:/Users/sonbu/Desktop/LVTN/3D_Model/demo_LIDC.txt'
label_file = 'C:/Users/sonbu/Desktop/LVTN/3D_Model/demo_label_LIDC.txt'
model_checkpoint = './Model/Unet/Unet_LIDC.pt'
features = [16, 32, 64, 128, 256]
img_size = 256
msk_size = 68#340

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print(device)

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = 3, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size = 3, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, inputs):
        return self.conv(inputs)

class Unet_1(nn.Module):
    def __init__(self,in_c,out_c):
        super().__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.down1 = DoubleConv(in_c,features[0])
        self.down2 = DoubleConv(features[0],features[1])
        self.down3 = DoubleConv(features[1],features[2])
        self.down4 = DoubleConv(features[2],features[3])
        self.down5 = DoubleConv(features[3],features[4])

        self.ConvT1 = nn.ConvTranspose2d(features[4],features[3], kernel_size=2, stride=2)
        self.ConvT2 = nn.ConvTranspose2d(features[3],features[2], kernel_size=2, stride=2)
        self.ConvT3 = nn.ConvTranspose2d(features[2],features[1], kernel_size=2, stride=2)
        self.ConvT4 = nn.ConvTranspose2d(features[1],features[0], kernel_size=2, stride=2)
        
        self.up1 = DoubleConv(features[4],features[3])
        self.up2 = DoubleConv(features[3],features[2])
        self.up3 = DoubleConv(features[2],features[1])
        self.up4 = DoubleConv(features[1],features[0])
        
        self.output = nn.Conv2d(features[0], out_c, kernel_size=1)
    
    def forward(self, inputs):
        d1 = self.down1(inputs)
        m1 = self.maxpool(d1)
        
        d2 = self.down2(m1)
        m2 = self.maxpool(d2)
        
        d3 = self.down3(m2)
        m3 = self.maxpool(d3)
        
        d4 = self.down4(m3)
        m4 = self.maxpool(d4)
        
        d5 = self.down5(m4)
    
        bottleneck = self.ConvT1(d5)
        c1 = TF.resize(d4,size = [bottleneck.shape[2],bottleneck.shape[2]])
        u1 = self.up1(torch.cat([bottleneck,c1],1))
        
        u1 = self.ConvT2(u1)
        c2 = TF.resize(d3,size = [u1.shape[2],u1.shape[2]])
        u2 = self.up2(torch.cat([u1,c2],1))

        u2 = self.ConvT3(u2)
        c3 = TF.resize(d2,size = [u2.shape[2],u2.shape[2]])
        u3 = self.up3(torch.cat([u2,c3],1))

        u3 = self.ConvT4(u3)
        c4 = TF.resize(d1,size = [u3.shape[2],u3.shape[2]])
        u4 = self.up4(torch.cat([u3,c4],1))

        output = self.output(u4)
        return torch.sigmoid(output)
        img_size = 128

dataset = []

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),])

dataset = []
f = open(data_file, "r")
for line in f:
    data = Image.open(line.replace('\n', ''))
    data = transform(data)
    data = data.unsqueeze(0)
    dataset.append(data)

dataloader = torch.cat(dataset, dim=0).to(device)

model = Unet_1(1,1).to(device)
model.load_state_dict(torch.load(model_checkpoint))

pred = model(dataloader)

X = dataloader.squeeze().cpu().detach().numpy()
Y = pred.squeeze().cpu().detach().numpy()

X = (np.array(X) * 255).astype(np.uint8)
Y = (np.array(Y) * 255).astype(np.uint8)
Y_pred = np.zeros(X.shape)
for i in range(len(Y)):
    Y_pred[i] = cv2.resize(Y[i], (img_size, img_size))


plt.figure(figsize=(20, 5))
num_plot = 10
for i in range(len(X)):
    if i < num_plot:
        plt.subplot(2, num_plot, i + 1)
        plt.imshow(X[i])
        plt.subplot(2, num_plot, num_plot + i + 1)
        plt.imshow(Y_pred[i],cmap='gray')
plt.show()

p = X.transpose(2,1,0)
# tái tạo 3d
verts, faces, normals, values = measure.marching_cubes(p, 20)
# đưa về tensor của open3d
mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(verts * np.array([1., 1., img_size / 50])), o3d.utility.Vector3iVector(faces))
# màu
mesh.paint_uniform_color(np.array([64,224,208])/255)
# lọc
mesh = mesh.filter_smooth_taubin(number_of_iterations=15)
mesh.compute_vertex_normals()

# trong suốt
mat = o3d.visualization.rendering.MaterialRecord()
mat.shader = "defaultLitTransparency"
mat.base_color = np.array([1, 1, 1, .5])

p = Y_pred.transpose(2,1,0)
verts, faces, normals, values = measure.marching_cubes(p, 20)

mesh1 = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(verts * np.array([1., 1., img_size / 50])), o3d.utility.Vector3iVector(faces))
mesh1.paint_uniform_color(np.array([224,64,208])/255)
mesh1 = mesh1.filter_smooth_taubin(number_of_iterations=20)
mesh1.compute_vertex_normals()

mat1 = o3d.visualization.rendering.MaterialRecord()
mat1.shader = "defaultLitTransparency"
mat1.base_color = np.array([1, 1, 1, .8])

import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

gui.Application.instance.initialize()
w = o3d.visualization.O3DVisualizer('3D model', 1080, 720)
w.set_background([1.0, 1.0, 1.0, 1.0], None)
w.add_geometry('brain', mesh, mat)
w.add_geometry('tumor', mesh1, mat1)
w.reset_camera_to_default()
gui.Application.instance.add_window(w)
gui.Application.instance.run()




