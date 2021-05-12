import math
import numpy as np
from typing import List


def rotate_matrix(theta: float, axis: str,  radian: bool = False):
    '''返回某个轴的旋转矩阵'''
    if not radian:
        theta = theta/180*np.pi
    assert axis in ['x', 'y', 'z'], "axis should in ['x','y','z']"
    if axis == 'x':
        return np.matrix([[1, 0, 0], [0, math.cos(theta), -math.sin(theta)], [0, math.sin(theta), math.cos(theta)]])
    elif axis == 'y':
        return np.matrix([[math.cos(theta), 0, math.sin(theta)], [0, 1, 0], [-math.sin(theta), 0, math.cos(theta)]])
    elif axis == 'z':
        return np.matrix([[math.cos(theta), -math.sin(theta), 0], [math.sin(theta), math.cos(theta), 0], [0, 0, 1]])


def reconstruct(left: np.ndarray, right: np.ndarray, alpha: List[float], beta: List[float], l: List[float], D: List[float]):
    """中心线的三维重建

    args:
        left: 左图的匹配点,维度为(N,2)
        right: 右图的匹配点,维度为(N,2)
        alpha: 左图与右图的LAO/RAO(光源围绕y轴旋转,与Z轴夹角),为角度值(-90~90)
        beta: 左图与右图的CRA/CAU(光源围绕X轴旋转,与Z轴夹角),为角度值(-90~90)
        l:光源与世界坐标系原点的距离
        D:光源与世界坐标系下两张图像的距离
    returns:
        points_3d:世界坐标系下,匹配点的坐标
    """
    # 计算旋转矩阵R: R = Rx(β2)·Ry(α2)·Ry(-α1)·Rx(-β1)
    Rx_b2 = rotate_matrix(beta[1], axis='x')
    Rx_b1 = rotate_matrix(-beta[0], axis='x')
    Ry_a2 = rotate_matrix(alpha[1], axis='y')
    Ry_a1 = rotate_matrix(-alpha[0], axis='y')
    R = Rx_b2*Ry_a2*Ry_a1*Rx_b1

    # 计算平移矢量t(3*1): t = T1-Rx(β1)·Ry(α1)·Ry(-α2)·Rx(-β2)·T2
    T1 = np.matrix([0, 0, l[0]]).T
    T2 = np.matrix([0, 0, l[1]]).T
    Rx_b2 = rotate_matrix(-beta[1], axis='x')
    Rx_b1 = rotate_matrix(beta[0], axis='x')
    Ry_a2 = rotate_matrix(-alpha[1], axis='y')
    Ry_a1 = rotate_matrix(alpha[0], axis='y')
    t = T1 - Rx_b1*Ry_a1*Ry_a2*Rx_b2*T2

    # 计算几何变换矩阵GT(3*4): GT = [R,t]
    GT = np.c_[R, t]

    # 计算投影平面上的点在两个光源的坐标系中的坐标
    # 真实世界中的点记为P(x,y,z),光源记为s1,s2
    # 点P在两个投影平面上的坐标为(u1,v1),(u2,v2)
    # 光源与真实点构成空间坐标系X1Y1Z1S1和X2Y2Z2S2,其中Z1和Z2与两个投影平面垂直

    length = len(left)
    # 计算ξ与η:
    # ξ1i = u1i/D1 = x1i/z1i,ξ2i = u2i/D2 = x2i/z2i
    # η1i = v1i/D1 = y1i/z1i,η2i = v2i/D2 = y2i/z2i
    # i∈[0,length)
    ksi1 = [u1/D[0] for u1 in left[:, 0]]
    eta1 = [v1/D[0] for v1 in left[:, 1]]
    ksi2 = [u2/D[1] for u2 in right[:, 0]]
    eta2 = [v2/D[1] for v2 in right[:, 1]]

    # 最小二乘法求解 A·C = B
    # C = (AT·A)-1·AT·B
    # 对每个匹配点分别求解
    C = []
    for i in range(length):
        # 计算a,b
        # a = [r11-r31·ξ2i,r12-r32·ξ2i,r13-r33·ξ2i]
        # b = [r21-r31·η2i,r22-r32·η2i,r23-r33·η2i]
        a = np.matrix([R[0, 0]-R[2, 0]*ksi2[i], R[0, 1]-R[2, 1]*ksi2[i], R[0, 2]-R[2, 2]*ksi2[i]])
        b = np.matrix([R[1, 0]-R[2, 0]*eta2[i], R[1, 1]-R[2, 1]*eta2[i], R[1, 2]-R[2, 2]*eta2[i]])
        at = a*t
        at = at[0,0] # 取出矩阵元素
        bt = b*t
        bt = bt[0,0] # 取出矩阵元素
        B = np.matrix([[0], [0], [at], [bt]])
        A = np.matrix([
            [1, 0, -ksi1[i]],
            [0, 1, -eta1[i]],
            [R[0, 0]-R[2, 0]*ksi2[i], R[0, 1]-R[2, 1]*ksi2[i], R[0, 2]-R[2, 2]*ksi2[i]],
            [R[1, 0]-R[2, 0]*eta2[i], R[1, 1]-R[2, 1]*eta2[i], R[1, 2]-R[2, 2]*eta2[i]]
        ])
        C.append((A.T * A).I * A.T * B)
    
    

if __name__ == '__main__':
    left = np.ones([14, 2])
    right = np.ones([14, 2])
    alpha = [40.1, -3.8]
    beta = [-5.1, 31]
    l = [776, 840]
    D = [990, 989]
    reconstruct(left, right, alpha, beta, l, D)

