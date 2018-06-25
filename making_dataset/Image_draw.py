#-*- coding: utf-8 -*-
import tkinter as tk
from PIL import ImageDraw, Image, ImageTk
import csv

class ExampleApp(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.x = self.y = 0
        self.img_num = 282
        self.img_url = '../dataset_180529/'+ str(self.img_num)+'.jpg'
        self.im = Image.open(self.img_url)
        self.canvas = tk.Canvas(self, width=self.im.size[0], height=self.im.size[1], cursor="cross")
        self.canvas.focus_set()
        self.canvas.pack(side="top", fill="both", expand=True)

        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_move_press)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)

        self.canvas.bind('<Left>', self.leftKey)
        self.canvas.bind('<Right>', self.rightKey)
        for i in range(9):
            i = i+1
            self.canvas.bind(i, self.on_pressed_num)

        self.rect = None

        self.start_x = None
        self.start_y = None

        self._draw_image()
        self._open_csv()

    def _open_csv(self):
        self.o = open('dataset.csv', 'a', encoding='euc_kr', newline='')
        self.wr = csv.writer(self.o)

    def leftKey(self, event):
        print("left Key pressed")
        self.img_num = self.img_num - 1
        self.im = Image.open('../dataset_180529/' + str(self.img_num) + '.jpg')
        self._draw_image()
        print(self.img_num)

    def rightKey(self, event):
        print("Right Key pressed")
        self.img_num = self.img_num + 1
        self.im = Image.open('../dataset_180529/' + str(self.img_num) + '.jpg')
        self._draw_image()
        print(self.img_num)

    def _draw_image(self):
         self.tk_im = ImageTk.PhotoImage(self.im)
         self.canvas.create_image(0,0, anchor="nw",image=self.tk_im)

    def on_pressed_num(self, event):
        print("X min : {}, X max : {},"
              "Y min : {}, Y max : {}"
              .format(self.start_x, curX, self.start_y, curY))
        if event.keysym == '1':
            print(event.keysym + " : 파프리카")
        elif event.keysym == '2':
            print(event.keysym + " : 달걀")
        elif event.keysym == '3':
            print(event.keysym + " : 오렌지")
        elif event.keysym == '4':
            print(event.keysym + " : 사과")
        elif event.keysym == '5':
            print(event.keysym + " : 토마토")
        elif event.keysym == '6':
            print(event.keysym + " : 코카콜라")
        elif event.keysym == '7':
            print(event.keysym + " : 펩시콜라")
        elif event.keysym == '8':
            print(event.keysym + " : 콩자반")
        elif event.keysym == '9':
            print(event.keysym + " : 발사믹 드레싱")
        list = []
        list.append(str(self.img_num) + '.jpg')
        list.append(self.start_x)
        list.append(curX)
        list.append(self.start_y)
        list.append(curY)
        list.append(event.keysym)
        self.wr.writerow(list)


    def on_button_press(self, event):
        # save mouse drag start position
        self.start_x = event.x
        self.start_y = event.y

        # create rectangle if not yet exist
        #if not self.rect:
        self.rect = self.canvas.create_rectangle(self.x, self.y, 1, 1)

    def on_move_press(self, event):
        global curX, curY
        curX, curY = (event.x, event.y)

        # expand rectangle as you drag the mouse
        self.canvas.coords(self.rect, self.start_x, self.start_y, curX, curY)

    def on_button_release(self, event):
        #print(self.start_x)
        #print(self.start_y)
        #print(curX)
        #print(curY)
        pass


if __name__ == "__main__":
    app = ExampleApp()
    app.mainloop()