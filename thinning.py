# -*- coding: utf-8 -*-

import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import copy

# Zhang Suen thining algorythm
def Zhang_Suen_thining(img):
    # get shape
    if img.ndim >2:
        H, W, C = img.shape
    else:
        H, W = img.shape

    # prepare out image
    out = np.zeros((H, W), dtype=np.int)
    #自定义变量，用于获取边界点
    eout = out
    etout = out
    out[img[..., 0] > 0] = 1

    # inverse
    out = 1 - out
    
    #自动变量初始化
    u=[]
    edge=[]
    edge_thin=[]

    n=1 #标识符，用于判断第一次细化
    while True:
        s1 = []
        s2 = []

        # step 1 
        for y in range(1, H-1):
            for x in range(1, W-1):
                # condition 1
                if out[y, x] > 0:
                    continue

                # condition 2
                f1 = 0
                if (out[y-1, x+1] - out[y-1, x]) == 1:
                    f1 += 1
                if (out[y, x+1] - out[y-1, x+1]) == 1:
                    f1 += 1
                if (out[y+1, x+1] - out[y, x+1]) == 1:
                    f1 += 1
                if (out[y+1, x] - out[y+1, x+1]) == 1:
                    f1 += 1
                if (out[y+1, x-1] - out[y+1, x]) == 1:
                    f1 += 1
                if (out[y, x-1] - out[y+1, x-1]) == 1:
                    f1 += 1
                if (out[y-1, x-1] - out[y, x-1]) == 1:
                    f1 += 1
                if (out[y-1, x] - out[y-1, x-1]) == 1:
                    f1 += 1

                if f1 != 1:
                    continue
                    
                # condition 3
                f2 = np.sum(out[y-1:y+2, x-1:x+2])
                
                if f2 > 1 and f2 <= 6 and n==1:
                    edge.append([y,x])                    
                    
                if f2 < 2 or f2 > 6:
                    continue
                
                # condition 4
                # x2 x4 x6
                if (out[y-1, x] + out[y, x+1] + out[y+1, x]) < 1 :
                    continue

                # condition 5
                # x4 x6 x8
                if (out[y, x+1] + out[y+1, x] + out[y, x-1]) < 1 :
                    continue
                    
                s1.append([y, x])
        
#        n=2
#        print('edgesize:',len(edge))
        
        for v in s1:
            out[v[0], v[1]] = 1
        
        #记录最外圈的轮廓点坐标
        if n==1 :   
            tests1=s1
        
        #记录边缘图像
        for e in edge:
            eout[e[0], e[1]] = 1
            
        # step 2 
        for y in range(1, H-1):
            for x in range(1, W-1):
                
                # condition 1
                if out[y, x] > 0:
                    continue

                # condition 2: A(P1)=1
                f1 = 0
                if (out[y-1, x+1] - out[y-1, x]) == 1:
                    f1 += 1
                if (out[y, x+1] - out[y-1, x+1]) == 1:
                    f1 += 1
                if (out[y+1, x+1] - out[y, x+1]) == 1:
                    f1 += 1
                if (out[y+1, x] - out[y+1, x+1]) == 1:
                    f1 += 1
                if (out[y+1, x-1] - out[y+1, x]) == 1:
                    f1 += 1
                if (out[y, x-1] - out[y+1, x-1]) == 1:
                    f1 += 1
                if (out[y-1, x-1] - out[y, x-1]) == 1:
                    f1 += 1
                if (out[y-1, x] - out[y-1, x-1]) == 1:
                    f1 += 1

                if f1 != 1:
                    continue
                    
                # condition 3
                f2 = np.sum(out[y-1:y+2, x-1:x+2])
                
                if f2 > 1 and f2 <= 6 :
                    edge_thin.append([y,x])
                
                if f2 < 2 or f2 > 6:
                    continue
                
                # condition 4
                # x2 x4 x8
                if (out[y-1, x] + out[y, x+1] + out[y, x-1]) < 1 :
                    continue

                # condition 5
                # x2 x6 x8
                if (out[y-1, x] + out[y+1, x] + out[y, x-1]) < 1 :
                    continue
                    
                s2.append([y, x])
               
#                n=n+1  
        
        for v in s2:
            out[v[0], v[1]] = 1
            
        if n==1: 
            tests2= s2    
            
#        for et in edge:
#            etout[et[0], et[1]] = 1
            
        for ed in edge_thin:
            etout[ed[0],ed[1]] = 1
            
        # if not any pixel is changed
        if len(s1) < 1 and len(s2) < 1:
            break
    
        n=n+1
        
    
    #for boundary        
    for i in range(len(out)-1):
        uy = out[i+1][0] - out[i][0]
        ux = out[i+1][1] - out[i][1]
        u.append([uy,ux])    
    
#    消除最左侧的白边(特殊图片处理)
    out[:,0] = 1
    eout[:,0] = 0
#    etout[:,0] = 0
    
    out = 1 - out    
    out = out.astype(np.uint8) * 255
    
#    外边界轮廓像素值
    eout = eout.astype(np.uint8) * 255
    etout = etout.astype(np.uint8) * 255
    
#    外边界坐标点--有误（画出的图为所有点的坐标）
    ar_ed=[]
    for ary1 in range(H):
        for arx1 in range(W):
            if eout[ary1,arx1] == 255:
                ar_ed.append([ary1,arx1])
#    
#    中心点坐标
    ar=[]
    for y1 in range(H):
        for x1 in range(W):
            if out[y1,x1] == 255:
                ar.append([y1,x1])
    
#    剔除中心线的外轮廓
    idx = out == 255    
    slc = eout
    slc[idx] = 0
    
#    return out,eout,ar,ar_ed,slc,edge_thin,tests1,tests2
    return out,eout,ar,ar_ed,slc,etout,tests1,tests2

#两幅输入图像，此处采用二值图而非原图，原图还需要另外处理
img1 = cv2.imread("d40_erzhi.png")
img2 = cv2.imread("d41_erzhi.png")


# out1 = Zhang_Suen_thining(img1)[0]    #细化的中心线
# #edge1 = Zhang_Suen_thining(img1)[1]    #外边界轮廓
# ar1 = Zhang_Suen_thining(img1)[2]   #中心坐标点
# ar_ed1 = Zhang_Suen_thining(img1)[3]    #所有坐标点
# #slc1 = Zhang_Suen_thining(img1)[4]  #剔除细化中心线的轮廓
# #edge_thin1=Zhang_Suen_thining(img1)[5]
# ts11 = Zhang_Suen_thining(img1)[6]  #细化步骤1最外圈的轮廓
# ts21 = Zhang_Suen_thining(img1)[7]  #细化步骤2最外圈的轮廓

out1,_,ar1,ar_ed1,_,_,ts11,ts21=Zhang_Suen_thining(img1)

# out2 = Zhang_Suen_thining(img2)[0]
# edge2 = Zhang_Suen_thining(img2)[1]
# ar2 = Zhang_Suen_thining(img2)[2]
# ar_ed2 = Zhang_Suen_thining(img2)[3]
# ts12 = Zhang_Suen_thining(img2)[6]
# ts22 = Zhang_Suen_thining(img2)[7]

out2,edge2,ar2,ar_ed2,_,_,ts12,ts22 = Zhang_Suen_thining(img2)
#中心线点的像素值
qx1=[]
qy1=[]

for i in range(len(out1)):
    qx1.append(out1[i][0])
    qy1.append(out1[i][1])
    
qx2=[]
qy2=[]
for j in range(len(out2)):
    qx2.append(out2[j][0])
    qy2.append(out2[j][1])

###坐标点
#lunkx1=[]
#lunky1=[]
#for l in range(len(edge_thin1)):
#    lunkx1.append(edge_thin1[l][0])
#    lunky1.append(edge_thin1[l][1])
#
#plt.scatter(lunkx1,lunky1,1)    

#所有点坐标
qx01=[]
qy01=[]
for i1 in range(len(ar_ed1)):
    qx01.append(ar_ed1[i1][0])
    qy01.append(ar_ed1[i1][1])
    
qx02=[]
qy02=[]
for i2 in range(len(ar_ed2)):
    qx02.append(ar_ed2[i2][0])
    qy02.append(ar_ed2[i2][1])

#画图
plt.scatter(qx01,qy01,1)
plt.scatter(qx02,qy02,1)

#判断是不是外轮廓点
tsres1 = copy.deepcopy(ts11)
for ts1 in ts21:
    if ts1 not in ts11:
        tsres1.append(ts1)
        
tsres2 = copy.deepcopy(ts12)
for ts2 in ts22:
    if ts2 not in ts12:
        tsres2.append(ts2)

#外轮廓点       
qxc1=[]
qyc1=[]
for c1 in range(len(tsres1)):
    qxc1.append(tsres1[c1][0])
    qyc1.append(tsres1[c1][1])    
qxc2=[]
qyc2=[]
for c2 in range(len(tsres2)):
    qxc2.append(tsres2[c2][0])
    qyc2.append(tsres2[c2][1])    

plt.scatter(qxc1,qyc1,1)
plt.scatter(qxc2,qyc2,1)


#针对一张图片进行的二维血管直径获取
cor=[]
cx=[]
cy=[]
#中心点坐标
for cc in range(len(ar1)):
    cx.append(ar1[cc][1])
    cy.append(ar1[cc][0])


#获取二维血管的直径(存在问题，需改正)
#根据前后中心点的斜率获取垂直直线的交点    
dis2D=[]
cor=[]
for ed in range(len(ar1)-1):
     
    kt =  - (cx[ed+1] - cx[ed])/(cy[ed+1] - cy[ed] +0.001)
    
    if abs(kt) <= 0.001 or abs(kt) > 999:
        cor.append([tsres1[ed][1],tsres1[ed][0]])
    elif kt > 1000:
        kidx = [ i for i, x in enumerate(qyc1) if x is cy[ed]]
        if len(kidx) == 0 :
            continue
        
        for i in range(len(kidx)-1):
            if kidx[i] < ed and kidx[i+1] >ed: 
                cor.append([qyc1[kidx[i]],qxc1[kidx[i]]])
            else:
                continue
    else:
        bt = cy[ed] - kt*cx[ed]
        for ctemp in tsres1:
            y_temp = kt*ctemp[1] + bt    
            if int(y_temp) == ctemp[0] and abs(ctemp[1]-cx[ed])<20:
                cor.append([ctemp[1],ctemp[0]])

    xx=[]
    yy=[]

    for pp in range(len(cor)):   
        if(abs(cor[pp][0]-cor[0][0]) < 50):
            xx.append(cor[pp][1])
            yy.append(cor[pp][0]) 
    
    #血管直径
    dis2D.append( math.sqrt((yy[-1] - yy[0])**2 + (xx[-1]-xx[0])**2))



'''
以下是中心线的三维重构
'''
#直接套用guozhuwang2011.pdf中给出的数据
a = [40.1,-3.8]  #造影角度 沿Y轴顺时针旋转
b = [-5.1,31]  #沿X轴顺时针旋转
D = [990,989]
L = [776,840]  #沿z轴平移距离
D1,D2 = 990,989



u1,v1 = qxc1,qyc1
u2,v2 = qxc2,qyc2

t=1
#参数矩阵
R_y1 = np.matrix([[math.cos(a[0]),0,-math.sin(a[0])],[0,1,0],[math.sin(a[0]),0,math.cos(a[0])]])
R_y2 = np.matrix([[math.cos(a[t]),0,-math.sin(a[t])],[0,1,0],[math.sin(a[t]),0,math.cos(a[t])]])
R_x1 = np.matrix([[1,0,0],[0,math.cos(b[0]),math.sin(b[0])],[0,-math.sin(b[0]),math.cos(b[0])]])
R_x2 = np.matrix([[1,0,0],[0,math.cos(b[t]),math.sin(b[t])],[0,-math.sin(b[t]),math.cos(b[t])]])

T1 = np.matrix([0,0,L[0]]).T
T2 = np.matrix([0,0,L[t]]).T

R_nx1 = np.matrix([[1,0,0],[0,math.cos(-b[0]),math.sin(-b[0])],[0,-math.sin(-b[0]),math.cos(-b[0])]])
R_nx2 = np.matrix([[1,0,0],[0,math.cos(-b[t]),math.sin(-b[t])],[0,-math.sin(-b[t]),math.cos(-b[t])]])
R_ny1 = np.matrix([[math.cos(-a[0]),0,-math.sin(-a[0])],[0,1,0],[math.sin(-a[0]),0,math.cos(-a[0])]])
R_ny2 = np.matrix([[math.cos(-a[t]),0,-math.sin(-a[t])],[0,1,0],[math.sin(-a[t]),0,math.cos(-a[t])]])


#旋转矩阵3x3 
R = R_x2 * R_y2 * R_ny1 * R_nx1

e = np.matrix([0,0,0,1])

homo_x1 = np.r_[np.c_[R_x1,[0,0,0]],e]
homo_x2 = np.r_[np.c_[R_nx2,[0,0,0]],e]
homo_y1 = np.r_[np.c_[R_y1,[0,0,0]],e]
homo_y2 = np.r_[np.c_[R_ny2,[0,0,0]],e]

#平移矢量3x1 
#py_t = T1 - homo_x1 * homo_y1 * homo_y2 * homo_x2 * T2
py_t = np.matrix(T1 - R.I * T2)
#几何变换矩阵3x4
GT = np.c_[R,py_t]


#投影面坐标 p1(u1,v1)  p2(u2,v2)
#外极线匹配
a1,a2,a3 = [],[],[]
for p in range(len(u1)):
    a1.append(R[0,0]*(u1[p]/D[0])+R[0,1]*(v1[p]/D[0])+R[0,2])
    a2.append(R[1,0]*(u1[p]/D[0])+R[1,1]*(v1[p]/D[0])+R[1,2])
    a3.append(R[2,0]*(u1[p]/D[0])+R[2,1]*(v1[p]/D[0])+R[2,2])

b1,b2,b3 = [],[],[]
for pp in range(len(u1)):
    b1.append(R[0,0]*py_t[0,0]+R[0,1]*py_t[1,0]+R[0,2]*py_t[2,0])
    b2.append(R[1,0]*py_t[0,0]+R[1,1]*py_t[1,0]+R[1,2]*py_t[2,0])
    b3.append(R[2,0]*py_t[0,0]+R[2,1]*py_t[1,0]+R[2,2]*py_t[2,0])
    


sgma2 = [sg2/D2 for sg2 in u2 ]
sita2 = [st2/D2 for st2 in v2 ]
#外极线L2方程
#sgma *(a3*b2-a2*b3) + sita * (a1*b3-a3*b1) + (a2*b1-a1*b2) =0

#投影点p2(u2,v2)到外极线L2的距离
#d = abs((a3*b2-a2*b3)*u2 + (a1*b3-a3*b1)*v2 + (a2*b1-a1*b2)) / math.sqrt(pow((a3*b2-a2*b3),2) + pow((a1*b3-a3*b1),2))
    

sgma1 = [sg1/D2 for sg1 in u1 ]
sita1 = [st1/D2 for st1 in v1 ]

C=[]
for n in range(min(len(u1),len(u2))):   
    A = np.matrix([[1,0,-sgma1[n]],
              [0,1,-sita1[n]],
              [R[0,0]-R[2,0]*(sgma2[n]), R[0,1]-R[2,1]*(sgma2[n]), R[0,2]-R[2,2]*(sgma2[n])],
              [R[1,0]-R[2,0]*(sita2[n]), R[1,1]-R[2,1]*(sita2[n]), R[1,2]-R[2,2]*(sita2[n])]]) 

    a_vec= np.matrix([R[0,0]-R[2,0]*(sgma2[n]), R[0,1]-R[2,1]*(sgma2[n]), R[0,2]-R[2,2]*(sgma2[n])]) 
    b_vec= np.matrix([R[1,0]-R[2,0]*(sita2[n]), R[1,1]-R[2,1]*(sita2[n]), R[1,2]-R[2,2]*(sita2[n])]) 
    
    ta =  np.mat(a_vec*py_t)
    tb =  b_vec*py_t
    B = np.matrix([[0],[0],[ta[0,0]],[tb[0,0]]]) 

    C.append( (A.T * A).I * A.T * B )

x1,y1,z1 = [],[],[]
for ii in range(len(C)):
    x1.append(C[ii][0,0])
    y1.append(C[ii][1,0])
    z1.append(C[ii][2,0])


#射线源坐标(x1,y1,z1)与(x2,y2,z2)之间转换
#tp = np.matrix([x1,y1,z1]).T
#x2,y2,z2 = [float(x) for x in np.array(R*(tp - py_t))]

x2,y2,z2 = [],[],[]
for jj in range(len(x1)):   
    tp = np.matrix([x1[jj],y1[jj],z1[jj]]).T
    
    x2.append(np.array(R*(tp - py_t))[0,0])
    y2.append(np.array(R*(tp - py_t))[1,0])
    z2.append(np.array(R*(tp - py_t))[2,0])
    
    

#射线源坐标(x2,y2,z2)与世界坐标系坐标转换
homo_T2 = np.r_[np.c_[R_x1,[0,0,0]],e]

#shijie = np.matrix([x2,y2,z2,1]) * homo_T2.I * homo_x2 * homo_y2
#sj = shijie[0,:3]

sj=[]
for ss in range(len(x2)):
    shijie = np.matrix([x2[ss],y2[ss],z2[ss],1]) * homo_T2.I * homo_x2 * homo_y2
    sj.append(np.array(shijie[0,:3]))

sj_point=[]
for s in sj:
    sj_point.append(s[0])

#输出三维坐标点
outx,outy,outz=[],[],[]
for o in range(len(sj_point)):
    outx.append( sj_point[o][0])
    outy.append( sj_point[o][1])
    outz.append( sj_point[o][2])
    

#三维血管真实直径
dis3D=[]
for dd in range(len(outx)):    
    dis3D.append( dis2D * math.sqrt(outx[dd]**2 + outy[dd]**2 + outz[dd]**2)/math.sqrt(u1[dd]**2 + v1[dd]**2 + D1**2) )


outxy = np.c_[outx,outy]
rout = np.c_[np.c_[outx,outy],outz]
fxy = np.savetxt('outxy.txt',outxy,'%0.2f')

