import cv2
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List
from torchviz import make_dot
from mpl_toolkits.mplot3d import Axes3D

class RotateMatrix(torch.nn.Module):
    def __init__(self, axis: str) -> None:
        super().__init__()
        self.axis = axis

    def forward(self, x: float):
        theta = x/180 * \
            torch.tensor([np.pi], dtype=torch.float32, requires_grad=False)
        assert self.axis in ['x', 'y', 'z'], "axis should in ['x','y','z']"
        if self.axis == 'x':
            return torch.Tensor([[1, 0, 0], [0, torch.cos(theta), -torch.sin(theta)], [0, torch.sin(theta), torch.cos(theta)]])
        elif self.axis == 'y':
            return torch.Tensor([[torch.cos(theta), 0, torch.sin(theta)], [0, 1, 0], [-torch.sin(theta), 0, torch.cos(theta)]])
        elif self.axis == 'z':
            return torch.Tensor([[torch.cos(theta), -torch.sin(theta), 0], [torch.sin(theta), torch.cos(theta), 0], [0, 0, 1]])


class Reconstruct(torch.nn.Module):
    def __init__(self, alpha: List[float], beta: List[float], l: List[float], D: List[float]) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta

        # 计算从左图映射回空间坐标系X1Y1Z1S1的旋转矩阵R
        Rx_b2 = RotateMatrix('x')(beta[1])
        Rx_b1 = RotateMatrix('x')(-beta[0])
        Ry_a2 = RotateMatrix('y')(alpha[1])
        Ry_a1 = RotateMatrix('y')(-alpha[0])
        R = Rx_b2@Ry_a2@Ry_a1@Rx_b1

        # 计算平移矩阵t
        self.T1 = torch.Tensor([[0, 0, l[0]]]).T
        self.T2 = torch.Tensor([[0, 0, l[1]]]).T
        Rx_b2 = RotateMatrix('x')(-beta[1])
        Rx_b1 = RotateMatrix('x')(beta[0])
        Ry_a2 = RotateMatrix('y')(-alpha[1])
        Ry_a1 = RotateMatrix('y')(alpha[0])
        t = self.T1 - Rx_b1@Ry_a1@Ry_a2@Rx_b2@self.T2

        # 从X1Y1Z1S1映射回XYZO的旋转矩阵R_inv
        Ry_a1 = RotateMatrix('y')(-alpha[0])
        Rx_b1 = RotateMatrix('x')(-beta[0])
        self.R_inv = Ry_a1@Rx_b1

        # 从XYZO映射回u1v1的旋转矩阵R_inv1
        Rx_b = RotateMatrix('x')(beta[0])
        Ry_a = RotateMatrix('y')(alpha[0])
        self.R_inv1 = Rx_b@Ry_a
        # 从XYZO映射回u2v2的旋转矩阵R_inv2
        Rx_b = RotateMatrix('x')(beta[1])
        Ry_a = RotateMatrix('y')(alpha[1])
        self.R_inv2 = Rx_b@Ry_a

        self.l = torch.Tensor(l)
        self.D = torch.Tensor(D)
        # l.requires_grad_()
        # D.requires_grad_()
        R.requires_grad_()
        t.requires_grad_()
        # self.l = torch.nn.Parameter(l)
        # self.D = torch.nn.Parameter(D)
        self.R = torch.nn.Parameter(R)
        self.t = torch.nn.Parameter(t)
        # self.register_parameter("l", self.l)
        # self.register_parameter("D", self.D)
        self.register_parameter("R", self.R)
        self.register_parameter("t", self.t)

    def forward(self, p1, p2):
        length = len(p1)
        output = torch.zeros([length, 3])
        pred2 = torch.zeros([length, 2])
        pred1 = torch.zeros([length, 2])
        for i in range(length):
            ksi1 = p1[i, 0]/self.D[0]
            eta1 = p1[i, 1]/self.D[0]
            ksi2 = p2[i, 0]/self.D[1]
            eta2 = p2[i, 1]/self.D[1]
            a = torch.tensor([
                self.R[0, 0]-self.R[2, 0]*ksi2,
                self.R[0, 1]-self.R[2, 1]*ksi2,
                self.R[0, 2]-self.R[2, 2]*ksi2
            ])
            b = torch.Tensor([
                self.R[1, 0]-self.R[2, 0]*eta2,
                self.R[1, 1]-self.R[2, 1]*eta2,
                self.R[1, 2]-self.R[2, 2]*eta2
            ])

            at = a@self.t
            bt = b@self.t

            B = torch.Tensor([[0, 0, at, bt]]).T
            A = torch.Tensor([
                [1, 0, -ksi1],
                [0, 1, -eta1],
                a,
                b
            ])
            C = (A.T@A).inverse()@A.T@B
            temp = self.R_inv@(C-self.T1)
            temp_inv1 = self.R_inv1@temp+self.T1
            temp_inv2 = self.R_inv2@temp+self.T2
            output[i] = temp.squeeze()
            pred1[i, 0] = temp_inv1[0]/temp_inv1[2]*self.D[0]
            pred1[i, 1] = temp_inv1[1]/temp_inv1[2]*self.D[0]
            pred2[i, 0] = temp_inv2[0]/temp_inv2[2]*self.D[1]
            pred2[i, 1] = temp_inv2[1]/temp_inv2[2]*self.D[1]
        return pred1, pred2, output


class Loss(torch.nn.Module):
    def __init__(self, p1: torch.Tensor, p2: torch.Tensor) -> None:
        super().__init__()
        self.p1 = p1
        self.p2 = p2
        self.mse = torch.nn.MSELoss()

    def forward(self, pred1: torch.Tensor, pred2: torch.Tensor) -> torch.Tensor:
        """pred:(N,3)"""
        loss = self.mse(pred1, self.p1)+self.mse(pred2, self.p2)
        return loss


if __name__ == '__main__':
    img1 = cv2.imread(sys.path[0]+'\\data\\d40.png', 0)
    img2 = cv2.imread(sys.path[0]+'\\data\\d41.png', 0)
    left = np.loadtxt(sys.path[0]+'\\data\\p1.txt')
    right = np.loadtxt(sys.path[0]+'\\data\\p2.txt')
    alpha = [40.1, -3.8]
    beta = [-5.1, 31]
    l = [776, 840]
    D = [990, 989]
    model = Reconstruct(alpha, beta, l, D)
    p1 = torch.Tensor(left)
    p2 = torch.Tensor(right)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = Loss(p1, p2)
    pred1, pred2, output = model(p1, p2)



    output = output.detach().numpy()
    x = output[:,0]
    y = output[:,1]
    z = output[:,2]
    ax = plt.gca(projection='3d')
    ax.invert_zaxis()
    ax.scatter(x, y, z)

    pred1,pred2 = pred1.detach().numpy(),pred2.detach().numpy()
    u1,v1 = pred1[:,0],pred1[:,1]
    u2,v2 = pred2[:,0],pred2[:,1]
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(img1, 'gray')
    ax[0].plot(left[:, 0], left[:, 1], c='r')
    ax[0].plot(u1, v1, c='cyan')
    ax[1].imshow(img2, 'gray')
    ax[1].plot(right[:, 0], right[:, 1], c='r')
    ax[1].plot(u2, v2, c='cyan')
    plt.show()