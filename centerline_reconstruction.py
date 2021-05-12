import math
from thinning import R_y2
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
    T1 = np.matrix([0,0,l[0]]).T
    T2 = np.matrix([0,0,l[1]]).T
    Rx_b2 = rotate_matrix(-beta[1], axis='x')
    Rx_b1 = rotate_matrix(beta[0], axis='x')
    Ry_a2 = rotate_matrix(-alpha[1], axis='y')
    Ry_a1 = rotate_matrix(alpha[0], axis='y')
    t = T1 - Rx_b1*Ry_a1*R_y2*Rx_b2*T2


if __name__ == '__main__':
    left = np.ones([14,2])
    right = np.ones([14,2])
    alpha = [40.1,-3.8]
    beta = [-5.1,31]
