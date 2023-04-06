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

data_file = 'C:/Users/sonbu/Desktop/LVTN/3D_Model/demo_LiTS17.txt'
label_file = 'C:/Users/sonbu/Desktop/LVTN/3D_Model/demo_label_LiTS17.txt'
model_checkpoint = 'C:/Users/sonbu/Desktop/LVTN/Model/GCNN/GCNN_LiTS17.pt'
img_size = 68


device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print(device)

def create_adj_matrix(size):
    
    col, row = np.meshgrid(np.arange(size), np.arange(size))
    coord = np.stack((col, row), axis=2).reshape(-1, 2) / size
    dist = torch.from_numpy(cdist(coord, coord)).float().to(device)
    sigma_dist = torch.var(dist)

    A = torch.exp(-dist**2 / sigma_dist)
    A[A <= 0.00001] = 0
    
    D = torch.diag(torch.sum(A, axis=1))
    D_hat = torch.sqrt(torch.linalg.inv(D))

    A_hat =  torch.matmul(D_hat, torch.matmul(A, D_hat))
    
    return A_hat

adj = create_adj_matrix(size=img_size)

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.projection = nn.Linear(in_features, out_features, bias=False)

    def forward(self, input, adj):
        batch_size = input.size(0)
        support = self.projection(input)
        output = torch.stack([torch.mm(adj, support[b]) for b in range(batch_size)])
        return output

class GCNN(nn.Module):
    def __init__(self, img_size, in_features, adj):
        super().__init__()
        self.N = img_size ** 2
        self.adj = nn.Parameter(adj)
        self.gc1 = GraphConvolution(in_features, 4)
        self.gc2 = GraphConvolution(4, 3)
        self.gc3 = GraphConvolution(3, 1)

    def forward(self, x):
        x = F.leaky_relu(self.gc1(x, self.adj))
        x = F.leaky_relu(self.gc2(x, self.adj))
        x = self.gc3(x, self.adj)
        return torch.sigmoid(x)

dataset = []

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor()])

dataset = []
f = open(data_file, "r")
for line in f:
    data = Image.open(line.replace('\n', ''))
    data = transform(data)
    data = data.unsqueeze(0)
    dataset.append(data)

label = []
f = open(label_file, "r")
for line in f:
    data = Image.open(line.replace('\n', ''))
    data = transform(data)
    data = data.unsqueeze(0)
    label.append(data)

dataloader = torch.cat(dataset, dim=0).to(device)
label = torch.cat(label, dim=0).to(device)

model = GCNN(img_size=img_size, in_features=1, adj = adj).to(device)
model.load_state_dict(torch.load(model_checkpoint))

# Code rieng cua gcnn
batch_size = dataloader.size(0)
data = dataloader.permute(0, 2, 3, 1).view(batch_size, -1, 1)
pred = model(data)
pred = pred.view(batch_size, img_size, img_size, 1).permute(0, 3, 1, 2)
#threshold
pred[pred > 0.5] = 1
pred[pred <= 0.5] = 0
label[label > 0.5] = 1
label[label <= 0.5] = 0
###

X = dataloader.squeeze().cpu().detach().numpy()
Y_pred = pred.squeeze().cpu().detach().numpy()
Y_label = label.squeeze().cpu().detach().numpy()

X = (np.array(X) * 255).astype(np.uint8)
Y_pred = (np.array(Y_pred) * 255).astype(np.uint8)
Y_label = (np.array(Y_label) * 255).astype(np.uint8)
"""

plt.figure(figsize=(img_size, 5))
num_plot = X.shape[0]
for i in range(len(X)):
    if i < num_plot:
        plt.subplot(3, num_plot, i + 1)
        plt.imshow(X[i])
        plt.subplot(3, num_plot, num_plot + i + 1)
        plt.imshow(Y_label[i],cmap='gray')
        plt.subplot(3, num_plot, 2 * num_plot + i + 1)
        plt.imshow(Y_pred[i],cmap='gray')
plt.show()
"""
p = X.transpose(2,1,0)
# tái tạo 3d
verts, faces, normals, values = measure.marching_cubes(p, 20)
# đưa về tensor của open3d
mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(verts * np.array([1., 1., img_size / 300])), o3d.utility.Vector3iVector(faces))
# màu
mesh.paint_uniform_color(np.array([64,224,208])/255)
# lọc
mesh = mesh.filter_smooth_taubin(number_of_iterations=50)
mesh.compute_vertex_normals()

# trong suốt
mat = o3d.visualization.rendering.MaterialRecord()
mat.shader = "defaultLitTransparency"
mat.base_color = np.array([1, 1, 1, .5])

p = Y_pred.transpose(2,1,0)
verts, faces, normals, values = measure.marching_cubes(p, 20)

mesh1 = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(verts * np.array([1., 1., img_size / 300])), o3d.utility.Vector3iVector(faces))
mesh1.paint_uniform_color(np.array([224,64,208])/255)
mesh1 = mesh1.filter_smooth_taubin(number_of_iterations=50)
mesh1.compute_vertex_normals()

mat1 = o3d.visualization.rendering.MaterialRecord()
mat1.shader = "defaultLitTransparency"
mat1.base_color = np.array([1, 1, 1, .8])

p = Y_label.transpose(2,1,0)
verts, faces, normals, values = measure.marching_cubes(p, 20)

mesh2 = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(verts * np.array([1., 1., img_size / 300])), o3d.utility.Vector3iVector(faces))
mesh2.paint_uniform_color(np.array([208,224,64])/255)
mesh2 = mesh2.filter_smooth_taubin(number_of_iterations=50)
mesh2.compute_vertex_normals()

mat2 = o3d.visualization.rendering.MaterialRecord()
mat2.shader = "defaultLitTransparency"
mat2.base_color = np.array([1, 1, 1, .8])

import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

gui.Application.instance.initialize()
w = o3d.visualization.O3DVisualizer('3D model', 1600, 900)
w.set_background([1.0, 1.0, 1.0, 1.0], None)
w.add_geometry('data', mesh, mat)
w.add_geometry('pred', mesh1, mat1)
w.add_geometry('label', mesh2, mat2)
w.reset_camera_to_default()
gui.Application.instance.add_window(w)
gui.Application.instance.run()




