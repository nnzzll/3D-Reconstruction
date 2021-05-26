import sys
import torch
import numpy as np
import matplotlib.pyplot as plt 

from scipy.interpolate import CubicSpline
from mpl_toolkits.mplot3d import Axes3D
# from reconstruction import rotate_matrix,reconstruct,BackProjection
from medvision.math3d.reconstruction import SimulatedAnnealing,Reconstruct,BackProjection

def Spline(points:np.ndarray):
    t_ori = np.arange(1,len(points)+1)
    sp = CubicSpline(t_ori,points)
    t_new = np.arange(1,len(points),0.1)
    new_points = sp(t_new)

    return new_points


if __name__ == "__main__":
    arr = np.load(sys.path[0]+"\\data\\lvjinhui.npz")
    img1 = arr['img1']
    img2 = arr['img2']
    alpha = arr['alpha']
    beta = arr['beta']
    p1 = arr['p1']-400
    p2 = arr['p2']-400
    l = [720/0.2,720/0.2]
    D = [1122/0.2,1169/0.2]
    left = Spline(p1.copy())
    right = Spline(p2.copy())
    p1 = torch.Tensor(left)
    p2 = torch.Tensor(right)
    new_alpha, new_beta, new_l, new_D = SimulatedAnnealing(
        p1, p2, alpha.copy(), beta.copy(), l.copy(), D.copy(),1000)
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
    ax[0].plot(left[:, 0]+400, left[:, 1]+400, c='r')
    ax[0].plot(u1+400, v1+400, c='cyan')
    ax[1].imshow(img2, 'gray')
    ax[1].plot(right[:, 0]+400, right[:, 1]+400, c='r')
    ax[1].plot(u2+400, v2+400, c='cyan')

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
