import os
import cv2
import torch
import pydicom
import numpy as np
import tkinter as tk

from PIL import ImageTk, Image
from tkinter import Frame, filedialog, ttk
from tkinter.messagebox import *

from medvision.algorithm.segmentation import SeedBinaryThreshold, GetContours
from medvision.math3d.curve import Spline
from medvision.math3d.visualize import PointCloudVTK
from medvision.math3d.reconstruction import Reconstruct, BackProjection, SimulatedAnnealing


class MyGUI:
    def __init__(self):
        self.dicom1 = None
        self.dicom2 = None
        self.alpha = [0, 0]
        self.beta = [0, 0]
        self.l = [0, 0]
        self.D = [0, 0]
        self.p1 = []
        self.p2 = []
        self.tkimg1 = None
        self.tkimg2 = None
        self.flag1 = False
        self.flag2 = False
        self.spline = False
        self.idx1 = 0
        self.idx2 = 0

        self.root = tk.Tk()
        self.frame1 = tk.Frame(self.root)
        self.frame2 = tk.Frame(self.root)
        self.frame3 = tk.Frame(self.root)
        self.frame1.pack()
        self.frame2.pack(fill=tk.X, expand=tk.YES)
        self.frame3.pack(fill=tk.X, expand=tk.YES)
        self.frame3.columnconfigure(0, weight=1)
        self.cv1 = tk.Canvas(self.frame1, height=800, width=800)
        self.cv2 = tk.Canvas(self.frame1, height=800, width=800)
        self.cv1.grid(row=0, column=0)
        self.cv2.grid(row=0, column=1)
        self.button1 = tk.Button(
            self.frame3, text='读取DICOM1', relief='groove', width=10, command=self.read1)
        self.button2 = tk.Button(
            self.frame3, text='读取DICOM2', relief='groove', width=10, command=self.read2)
        self.button3 = tk.Button(
            self.frame3, text="保存", relief='groove', width=10, command=self.save)
        self.button4 = tk.Button(
            self.frame3, text="加载", relief='groove', width=10, command=self.load)
        self.button5 = tk.Button(
            self.frame3, text="分割左图", relief='groove', width=10, command=self.seg_left)
        self.button6 = tk.Button(
            self.frame3, text="分割右图", relief='groove', width=10, command=self.seg_right)
        self.button7 = tk.Button(
            self.frame3, text="生成龙骨", relief='groove', width=10, command=self.centerline)
        self.button1.grid(row=0, column=1, padx=3, pady=10, sticky=tk.E)
        self.button2.grid(row=0, column=2, padx=3, pady=10, sticky=tk.E)
        self.button3.grid(row=0, column=3, padx=3, pady=10, sticky=tk.E)
        self.button4.grid(row=0, column=4, padx=3, pady=10, sticky=tk.E)
        self.button5.grid(row=0, column=5, padx=3, pady=10, sticky=tk.E)
        self.button6.grid(row=0, column=6, padx=3, pady=10, sticky=tk.E)
        self.button7.grid(row=0, column=0, padx=10, pady=10, sticky=tk.W)

        self.frame_idx1 = tk.IntVar()
        self.frame_idx1.set(1)
        self.frame_idx2 = tk.IntVar()
        self.frame_idx2.set(1)

    def save(self):
        path = filedialog.asksaveasfilename(defaultextension='.npz')
        if path:
            p1 = np.array(self.p1)
            p2 = np.array(self.p2)
            alpha = np.array(self.alpha)
            beta = np.array(self.beta)
            np.savez(
                path, p1=p1, p2=p2, alpha=alpha, beta=beta,
                img1=self.dicom1.pixel_array[self.idx1],
                img2=self.dicom2.pixel_array[self.idx2]
            )
            showinfo("", "保存成功！")

    def load(self):
        if self.flag1 and self.flag2:
            path = filedialog.askopenfilename()
            if path:
                if path.split('.')[-1] == 'npz':
                    try:
                        arr = np.load(path)
                        p1 = arr['p1']
                        p2 = arr['p2']
                        self.p1 = p1.tolist()
                        self.p2 = p2.tolist()
                    except:
                        showerror('', "文件错误！")
                else:
                    showinfo("提示", "无法读取的文件类型！")
        else:
            showinfo("提示", "请先读取数据！")

    def centerline(self):
        p1 = torch.Tensor(Spline(np.array(self.p1)))
        p2 = torch.Tensor(Spline(np.array(self.p2)))
        with torch.no_grad():
            new_alpha, new_beta, new_l, new_D = SimulatedAnnealing(
                p1, p2, self.alpha.copy(), self.beta.copy(), self.l.copy(), self.D.copy())
            xyz = Reconstruct(p1, p2, new_alpha, new_beta, new_l, new_D)
        output = xyz.detach().numpy()
        PointCloudVTK(output)

    def seg_left(self):
        idx = self.frame_idx1.get()-1
        sub = subWindow(
            self.root, self, self.dicom1.pixel_array[idx], np.array(self.p1))
        # sub.root.mainloop()
        self.root.wait_window(sub.root)

    def seg_right(self):
        idx = self.frame_idx2.get()-1
        sub = subWindow(
            self.root, self, self.dicom2.pixel_array[idx], np.array(self.p2))
        # sub.root.mainloop()
        self.root.wait_window(sub.root)

    def frame_select(self, text):
        self.show_frame()

    def show_frame(self):
        if self.flag1:
            idx = self.frame_idx1.get()-1
            if not self.p1:
                self.tkimg1 = ImageTk.PhotoImage(
                    image=Image.fromarray(self.dicom1.pixel_array[idx]))
            else:
                img = np.expand_dims(
                    self.dicom1.pixel_array[idx], axis=2).repeat(3, axis=2)
                if self.spline and len(self.p1) >= 2:
                    new_p1 = Spline(self.p1, dtype=int)
                    for p in new_p1:
                        img = cv2.circle(img, tuple(p), 2, (255, 0, 0), -1)
                    self.idx1 = idx
                else:
                    for p in self.p1:
                        img = cv2.circle(img, tuple(p), 2, (255, 0, 0), -1)
                self.tkimg1 = ImageTk.PhotoImage(image=Image.fromarray(img))
            self.cv1.create_image(0, 0, anchor='nw', image=self.tkimg1)
            self.cv1.update()

        if self.flag2:
            idx = self.frame_idx2.get()-1
            if not self.p2:
                self.tkimg2 = ImageTk.PhotoImage(
                    image=Image.fromarray(self.dicom2.pixel_array[idx]))
            else:
                img = np.expand_dims(
                    self.dicom2.pixel_array[idx], axis=2).repeat(3, axis=2)
                if self.spline and len(self.p2) >= 2:
                    new_p2 = Spline(self.p2, dtype=int)
                    for p in new_p2:
                        img = cv2.circle(img, tuple(p), 2, (0, 255, 255), -1)
                    self.idx2 = idx
                else:
                    for p in self.p2:
                        img = cv2.circle(img, tuple(p), 2, (0, 255, 255), -1)
                self.tkimg2 = ImageTk.PhotoImage(image=Image.fromarray(img))
            self.cv2.create_image(0, 0, anchor='nw', image=self.tkimg2)
            self.cv2.update()

    def read1(self):
        path = filedialog.askopenfilename()
        if(path):
            self.dicom1 = pydicom.read_file(path)
            self.flag1 = True
            self.alpha[0] = float(self.dicom1[0x0018, 0x1510].value)
            self.beta[0] = float(self.dicom1[0x0018, 0x1511].value)
            self.l[0] = float(self.dicom1[0x0018, 0x1110].value)
            self.D[0] = float(self.dicom1[0x0018, 0x1111].value)
            self.Scale1 = tk.Scale(
                self.frame2, length=800, orient=tk.HORIZONTAL, from_=1, to=len(self.dicom1.pixel_array), resolution=1,
                show=0, variable=self.frame_idx1, command=self.frame_select)
            self.Scale1.grid(row=0, column=0, padx=10)
            self.sp1 = tk.Spinbox(self.frame2, from_=1, to=len(self.dicom1.pixel_array),
                                  increment=1, textvariable=self.frame_idx1, command=self.show_frame)
            self.sp1.grid(row=1, column=0, sticky="W", padx=10, pady=10)
            self.cv1.bind("<Button-1>", self.get_points1)
            self.cv1.bind("<Button-2>", self.on_middle)
            self.cv1.bind("<Button-3>", self.pop_points1)
            self.show_frame()

    def read2(self):
        path = filedialog.askopenfilename()
        if(path):
            self.dicom2 = pydicom.read_file(path)
            self.flag2 = True
            self.alpha[1] = float(self.dicom2[0x0018, 0x1510].value)
            self.beta[1] = float(self.dicom2[0x0018, 0x1511].value)
            self.l[1] = float(self.dicom2[0x0018, 0x1110].value)
            self.D[1] = float(self.dicom2[0x0018, 0x1111].value)
            self.Scale2 = tk.Scale(
                self.frame2, length=800, orient=tk.HORIZONTAL, from_=1, to=len(self.dicom1.pixel_array),
                resolution=1, show=0, variable=self.frame_idx2, command=self.frame_select)
            self.Scale2.grid(row=0, column=1, padx=10)
            self.sp2 = tk.Spinbox(self.frame2, from_=1, to=len(self.dicom2.pixel_array),
                                  increment=1, textvariable=self.frame_idx2, command=self.show_frame)
            self.sp2.grid(row=1, column=1, sticky="W", padx=10, pady=10)
            self.cv2.bind("<Button-1>", self.get_points2)
            self.cv2.bind("<Button-2>", self.on_middle)
            self.cv2.bind("<Button-3>", self.pop_points2)
            self.show_frame()

    def get_points1(self, event):
        if self.flag1:
            self.p1.append([event.x, event.y])
            self.show_frame()

    def pop_points1(self, event):
        if self.flag1:
            if self.p1:
                self.p1.pop()
                self.show_frame()

    def get_points2(self, event):
        if self.flag2:
            self.p2.append([event.x, event.y])
            self.show_frame()

    def pop_points2(self, event):
        if self.flag2:
            if self.p2:
                self.p2.pop()
                self.show_frame()

    def on_middle(self, event):
        if not self.spline:
            self.spline = True
        else:
            self.spline = False
        self.show_frame()


class subWindow:
    def __init__(self, master, parent, img, point) -> None:
        self.root = tk.Toplevel(master)
        self.parent = parent
        self.img = img
        self.point = point
        self.click = True
        self.oval = np.zeros_like(img)
        self.mask = np.zeros_like(img)

        self.frame1 = tk.Frame(self.root)
        self.frame2 = tk.Frame(self.root)
        self.frame1.pack()
        self.frame2.pack(fill=tk.X, expand=tk.YES)
        self.frame2.columnconfigure(0, weight=1)
        self.cv1 = tk.Canvas(self.frame1, height=800, width=800)
        self.cv2 = tk.Canvas(self.frame1, height=800, width=800)
        self.cv1.grid(row=0, column=0)
        self.cv2.grid(row=0, column=1)
        self.tkimg1 = ImageTk.PhotoImage(image=Image.fromarray(img))
        self.tkimg2 = ImageTk.PhotoImage(
            image=Image.fromarray(np.zeros_like(img)))
        self.cv1.create_image(0, 0, anchor='nw', image=self.tkimg1)
        self.cv2.create_image(0, 0, anchor='nw', image=self.tkimg2)
        self.cv1.update()
        self.cv2.update()
        self.cv1.bind("<Button-2>", self.on_middle)
        self.button1 = tk.Button(
            self.frame2, text='自动分割', relief='groove', command=self.segment)
        self.button1.grid(row=0, column=0, padx=10, pady=10, sticky=tk.E)

    def on_middle(self, event):
        if self.click:
            self.click = False
            self.show()
        else:
            self.click = True
            self.show()

    def segment(self):
        img = cv2.GaussianBlur(self.img, (3, 3), 0)
        points = Spline(self.point)
        self.mask = SeedBinaryThreshold(img, points.astype(int), (24, 24))
        self.tkimg2 = ImageTk.PhotoImage(image=Image.fromarray(self.mask))
        self.c1, self.c2 = GetContours(self.point.astype(int), self.mask)
        self.parent.c1 = self.c1
        self.show()

    def show(self):
        contour_1 = Spline(self.c1, dtype=int)
        contour_2 = Spline(self.c2, dtype=int)
        img = cv2.cvtColor(self.img, cv2.COLOR_GRAY2RGB)
        for i in range(len(contour_1)):
            cv2.circle(
                img, (contour_1[i, 0], contour_1[i, 1]), 1, (255, 0, 0), -1)
            cv2.circle(
                img, (contour_2[i, 0], contour_2[i, 1]), 1, (0, 255, 0), -1)
        if self.click:
            self.tkimg1 = ImageTk.PhotoImage(image=Image.fromarray(img))
        else:
            self.tkimg1 = ImageTk.PhotoImage(image=Image.fromarray(self.img))
        self.cv1.create_image(0, 0, anchor='nw', image=self.tkimg1)
        if self.click:
            self.oval = np.zeros_like(self.img)
            c1 = self.c1.astype(int)
            c2 = self.c2.astype(int)
            for i in range(len(c1)):
                x = max(min(c1[i, 0], self.img.shape[1]-3), 3)
                y = max(min(c1[i, 1], self.img.shape[0]-3), 3)
                self.cv1.create_oval(x-3, y-3, x+3, y+3, fill='white')
                self.oval[y-3:y+3, x-3:x+3] = i+1
            for i in range(len(c2)):
                x = max(min(c2[i, 0], self.img.shape[1]-3), 3)
                y = max(min(c2[i, 1], self.img.shape[0]-3), 3)
                self.cv1.create_oval(x-3, y-3, x+3, y+3, fill='white')
                self.oval[y-3:y+3, x-3:x+3] = i+1+len(c1)
        self.cv2.create_image(0, 0, anchor='nw', image=self.tkimg2)
        self.cv1.bind("<ButtonPress-1>", self.on_press)
        self.cv1.update()
        self.cv2.update()

    def on_press(self, event):
        if self.oval[event.y, event.x]:
            self.idx = self.oval[event.y, event.x]
            self.cv1.bind("<B1-Motion>", self.on_move)

    def on_move(self, event):
        if self.idx <= len(self.point):
            self.c1[self.idx-1, 0] = event.x
            self.c1[self.idx-1, 1] = event.y
        else:
            self.c2[self.idx-len(self.point)-1, 0] = event.x
            self.c2[self.idx-len(self.point)-1, 1] = event.y
        self.show()


gui = MyGUI()
gui.root.mainloop()
