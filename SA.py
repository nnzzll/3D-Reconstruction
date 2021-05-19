'''A python implementation of simulated annealing alogorithm'''
import cv2
import sys
from numpy.core.numerictypes import maximum_sctype
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import Tensor


class RotateMatrix(torch.nn.Module):
    def __init__(self, axis: str) -> None:
        super().__init__()
        self.axis = axis

    def forward(self, x: float) -> Tensor:
        theta = x/180 * \
            torch.tensor([np.pi], dtype=torch.float32, requires_grad=False)
        assert self.axis in ['x', 'y', 'z'], "axis should in ['x','y','z']"
        if self.axis == 'x':
            return torch.Tensor([[1, 0, 0], [0, torch.cos(theta), -torch.sin(theta)], [0, torch.sin(theta), torch.cos(theta)]])
        elif self.axis == 'y':
            return torch.Tensor([[torch.cos(theta), 0, torch.sin(theta)], [0, 1, 0], [-torch.sin(theta), 0, torch.cos(theta)]])
        elif self.axis == 'z':
            return torch.Tensor([[torch.cos(theta), -torch.sin(theta), 0], [torch.sin(theta), torch.cos(theta), 0], [0, 0, 1]])


def Loss(pred: Tensor, target: Tensor) -> Tensor:
    return torch.norm(pred-target)


def Metroplis(E_new: Tensor, E_old: Tensor, T: Tensor) -> Tensor:
    if E_new < E_old:
        return torch.Tensor([1])
    else:
        return torch.exp(-(E_new-E_old)/T)


def Reconstruct(p1: Tensor, p2: Tensor, alpha: list, beta: list, l: list, D: list):
    # 计算从左图映射回空间坐标系X1Y1Z1S1的旋转矩阵R
    Rx_b2 = RotateMatrix('x')(beta[1])
    Rx_b1 = RotateMatrix('x')(-beta[0])
    Ry_a2 = RotateMatrix('y')(alpha[1])
    Ry_a1 = RotateMatrix('y')(-alpha[0])
    R = Rx_b2@Ry_a2@Ry_a1@Rx_b1

    # 计算平移矩阵t
    T1 = torch.Tensor([[0, 0, l[0]]]).T
    T2 = torch.Tensor([[0, 0, l[1]]]).T
    Rx_b2 = RotateMatrix('x')(-beta[1])
    Rx_b1 = RotateMatrix('x')(beta[0])
    Ry_a2 = RotateMatrix('y')(-alpha[1])
    Ry_a1 = RotateMatrix('y')(alpha[0])
    t = T1 - Rx_b1@Ry_a1@Ry_a2@Rx_b2@T2

    # 从X1Y1Z1S1映射回XYZO的旋转矩阵R_inv
    Ry_a1 = RotateMatrix('y')(-alpha[0])
    Rx_b1 = RotateMatrix('x')(-beta[0])
    R_inv = Ry_a1@Rx_b1

    # 计算
    length = len(p1)
    output = torch.zeros([length, 3])
    for i in range(length):
        ksi1 = p1[i, 0]/D[0]
        eta1 = p1[i, 1]/D[0]
        ksi2 = p2[i, 0]/D[1]
        eta2 = p2[i, 1]/D[1]
        a = torch.tensor([
            R[0, 0]-R[2, 0]*ksi2,
            R[0, 1]-R[2, 1]*ksi2,
            R[0, 2]-R[2, 2]*ksi2
        ])
        b = torch.Tensor([
            R[1, 0]-R[2, 0]*eta2,
            R[1, 1]-R[2, 1]*eta2,
            R[1, 2]-R[2, 2]*eta2
        ])

        at = a@t
        bt = b@t

        B = torch.Tensor([[0, 0, at, bt]]).T
        A = torch.Tensor([
            [1, 0, -ksi1],
            [0, 1, -eta1],
            a,
            b
        ])
        C = (A.T@A).inverse()@A.T@B
        temp = R_inv@(C-T1)
        output[i] = temp.squeeze()
    return output


def BackProjection(xyz: Tensor, alpha: list, beta: list, l: list, D: list):
    T1 = torch.Tensor([[0, 0, l[0]]]).T
    T2 = torch.Tensor([[0, 0, l[1]]]).T

    # 从XYZO映射回u1v1的旋转矩阵R_inv1
    Rx_b = RotateMatrix('x')(beta[0])
    Ry_a = RotateMatrix('y')(alpha[0])
    R_inv1 = Rx_b@Ry_a

    # 从XYZO映射回u2v2的旋转矩阵R_inv2
    Rx_b = RotateMatrix('x')(beta[1])
    Ry_a = RotateMatrix('y')(alpha[1])
    R_inv2 = Rx_b@Ry_a

    # 计算
    length = len(xyz)
    pred1 = torch.zeros([length, 2])
    pred2 = torch.zeros([length, 2])
    for i in range(length):
        temp = xyz[i].unsqueeze(1)
        temp_inv1 = R_inv1@temp+T1
        temp_inv2 = R_inv2@temp+T2
        pred1[i, 0] = temp_inv1[0]/temp_inv1[2]*D[0]
        pred1[i, 1] = temp_inv1[1]/temp_inv1[2]*D[0]
        pred2[i, 0] = temp_inv2[0]/temp_inv2[2]*D[1]
        pred2[i, 1] = temp_inv2[1]/temp_inv2[2]*D[1]
    return pred1, pred2


def SimulatedAnnealing(p1, p2, alpha, beta, L, D, T=250, T0=10, random_seed=2021):
    torch.manual_seed(random_seed)
    xyz = Reconstruct(p1, p2, alpha, beta, L, D)
    pred1, pred2 = BackProjection(xyz, alpha, beta, L, D)
    init_E = Loss(pred1, p1)+Loss(pred2, p2)
    E = 0
    E_old = init_E
    cnt = 0
    its = 0
    while(T > T0):
        a1 = alpha[0]-5 + 10*torch.rand(1)
        a2 = alpha[1]-5 + 10*torch.rand(1)
        b1 = beta[0]-5 + 10*torch.rand(1)
        b2 = beta[1]-5 + 10*torch.rand(1)
        l1 = L[0]-50 + 100*torch.rand(1)
        l2 = L[1]-50 + 100*torch.rand(1)
        d1 = D[0]-50 + 100*torch.rand(1)
        d2 = D[1]-50 + 100*torch.rand(1)
        new_xyz = Reconstruct(p1, p2, [a1, a2], [b1, b2], [l1, l2], [d1, d2])
        new_p1, new_p2 = BackProjection(
            new_xyz, [a1, a2], [b1, b2], [l1, l2], [d1, d2])
        E_new = Loss(new_p1, p1)+Loss(new_p2, p2)
        p = Metroplis(E_new, E_old, T)
        epsilon = torch.rand(1)
        print(epsilon, p)
        if epsilon < p:
            new_alpha = [a1, a2]
            new_beta = [b1, b2]
            new_L = [l1, l2]
            new_D = [d1, d2]
            cnt += 1
            E = E_new
        else:
            pass
        its += 1
        T = 0.9*T
    print("{}/{}".format(cnt, its))
    print("Init E:{}".format(init_E))
    print("New E:{}".format(E))
    return new_alpha, new_beta, new_L, new_D


if __name__ == '__main__':
    img1 = cv2.imread(sys.path[0]+'\\data\\d40.png', 0)
    img2 = cv2.imread(sys.path[0]+'\\data\\d41.png', 0)
    left = np.loadtxt(sys.path[0]+'\\data\\p1.txt')
    right = np.loadtxt(sys.path[0]+'\\data\\p2.txt')
    alpha = [40.1, -3.8]
    beta = [-5.1, 31]
    l = [776, 840]
    D = [990, 989]

    p1 = torch.Tensor(left)
    p2 = torch.Tensor(right)

    new_alpha, new_beta, new_l, new_D = SimulatedAnnealing(
        p1, p2, alpha.copy(), beta.copy(), l.copy(), D.copy())
    print(new_alpha, new_beta, new_l, new_D)

    xyz = Reconstruct(p1, p2, new_alpha, new_beta, new_l, new_D)
    pred1, pred2 = BackProjection(xyz, new_alpha, new_beta, new_l, new_D)

    output = xyz.detach().numpy()
    x = output[:, 0]
    y = output[:, 1]
    z = output[:, 2]
    ax = plt.gca(projection='3d')
    ax.invert_zaxis()
    ax.scatter(x, y, z)

    pred1, pred2 = pred1.detach().numpy(), pred2.detach().numpy()
    u1, v1 = pred1[:, 0], pred1[:, 1]
    u2, v2 = pred2[:, 0], pred2[:, 1]
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(img1, 'gray')
    ax[0].plot(left[:, 0], left[:, 1], c='r')
    ax[0].plot(u1, v1, c='cyan')
    ax[1].imshow(img2, 'gray')
    ax[1].plot(right[:, 0], right[:, 1], c='r')
    ax[1].plot(u2, v2, c='cyan')

    dist_err_1 = []
    dist_err_2 = []
    PixelSpacing = 0.2
    for i in range(len(left)):
        dist_err_1.append(
            PixelSpacing*np.sqrt(pow(u1[i]-left[i, 0], 2)+pow(v1[i]-left[i, 1], 2)))
        dist_err_2.append(
            PixelSpacing*np.sqrt(pow(u2[i]-right[i, 0], 2)+pow(v2[i]-right[i, 1], 2)))

    print("左图最大误差:{:.2f}mm".format(np.max(dist_err_1)))
    print("左图最小误差:{:.2f}mm".format(np.min(dist_err_1)))
    print("左图平均误差:{:.2f}mm".format(np.mean(dist_err_1)))
    print("左图误差方差:{:.2f}mm".format(np.var(dist_err_1)))

    print("右图最大误差:{:.2f}mm".format(np.max(dist_err_2)))
    print("右图最小误差:{:.2f}mm".format(np.min(dist_err_2)))
    print("右图平均误差:{:.2f}mm".format(np.mean(dist_err_2)))
    print("右图误差方差:{:.2f}mm".format(np.var(dist_err_2)))
    fig = plt.figure()
    plt.plot(np.arange(len(left)), dist_err_1, c='r', label='left_err')
    plt.plot(np.arange(len(left)), dist_err_2, c='cyan', label='right_err')
    plt.legend(loc='best')
    plt.show()
