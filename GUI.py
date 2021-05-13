import os
import cv2
import pydicom
import numpy as np
import tkinter as tk

from PIL import ImageTk, Image
from scipy.interpolate import CubicSpline
from tkinter import Frame, filedialog, ttk
from tkinter.messagebox import *


class MyGUI:
    def __init__(self):
        self.dicom1 = None
        self.dicom2 = None
        self.alpha = [0, 0]
        self.beta = [0, 0]
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
            self.frame3, text='读取DICOM1', relief='groove', command=self.read1)
        self.button2 = tk.Button(
            self.frame3, text='读取DICOM2', relief='groove', command=self.read2)
        self.button3 = tk.Button(
            self.frame3, text="保存", relief='groove', command=self.save)
        self.button1.grid(row=0, column=0, padx=10, pady=10, sticky=tk.E)
        self.button2.grid(row=0, column=1, padx=10, pady=10, sticky=tk.E)
        self.button3.grid(row=0, column=2, padx=10, pady=10, sticky=tk.E)
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
                    t_ori = np.arange(1, len(self.p1)+1)
                    sp = CubicSpline(t_ori, np.array(self.p1))
                    t_new = np.arange(1, len(self.p1), 0.1)
                    new_p1 = sp(t_new)
                    new_p1 = np.array(new_p1,dtype=int)
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
                    t_ori = np.arange(1, len(self.p2)+1)
                    sp = CubicSpline(t_ori, np.array(self.p2))
                    t_new = np.arange(1, len(self.p2), 0.1)
                    new_p2 = sp(t_new)
                    new_p2 = np.array(new_p2,dtype=int)
                    for p in new_p2:
                        img = cv2.circle(img, tuple(p), 2, (0, 255, 255), -1)
                    self.idx2 = idx
                else:
                    for p in self.p1:
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


gui = MyGUI()
gui.root.mainloop()
