import sys
import numpy as np
import matplotlib.pyplot as plt 

from scipy.interpolate import CubicSpline
from mpl_toolkits.mplot3d import Axes3D
from reconstruction import rotate_matrix,reconstruct,BackProjection


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
    p1 = arr['p1']
    p2 = arr['p2']
    l = [450,450]
    D = [1000,1000]
    p1 = Spline(p1.copy())
    p2 = Spline(p2.copy())
    x,y,z = reconstruct(p1,p2,alpha,beta,l,D)

    plt.figure()
    ax = plt.gca(projection='3d')
    ax.invert_zaxis()
    ax.scatter(x,y,z)
    

    u1,v1 = BackProjection(x,y,z,alpha[0],beta[0],500,1000)
    u2,v2 = BackProjection(x,y,z,alpha[1],beta[1],500,1000)
    fig,ax = plt.subplots(1,2)
    ax[0].imshow(img1,'gray')
    ax[0].plot(p1[:,0],p1[:,1],c='r')
    ax[0].plot(u1,v1,c='cyan')
    ax[1].imshow(img2,'gray')
    ax[1].plot(p2[:,0],p2[:,1],c='r')
    ax[1].plot(u2,v2,c='cyan')

    dist_err_1 = []
    dist_err_2 = []
    PixelSpacing = 0.2
    for i in range(len(p1)):
        dist_err_1.append(
            PixelSpacing*np.sqrt(pow(u1[i]-p1[i, 0], 2)+pow(v1[i]-p1[i, 1], 2)))
        dist_err_2.append(
            PixelSpacing*np.sqrt(pow(u2[i]-p2[i, 0], 2)+pow(v2[i]-p2[i, 1], 2)))

    print("左图最大误差:{:.2f}mm".format(np.max(dist_err_1)))
    print("左图最小误差:{:.2f}mm".format(np.min(dist_err_1)))
    print("左图平均误差:{:.2f}mm".format(np.mean(dist_err_1)))
    print("左图误差方差:{:.2f}mm".format(np.var(dist_err_1)))

    print("右图最大误差:{:.2f}mm".format(np.max(dist_err_2)))
    print("右图最小误差:{:.2f}mm".format(np.min(dist_err_2)))
    print("右图平均误差:{:.2f}mm".format(np.mean(dist_err_2)))
    print("右图误差方差:{:.2f}mm".format(np.var(dist_err_2)))
    fig = plt.figure()
    plt.plot(np.arange(len(p1)), dist_err_1, c='r', label='left_err')
    plt.plot(np.arange(len(p1)), dist_err_2, c='cyan', label='right_err')
    plt.legend(loc='best')
    plt.show()
