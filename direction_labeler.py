"""
A tool for efficiently labeling the direction of the fish.

Left key: Label current image as facing left.
Right key: Label current image as facing right.
"""

import tkinter as tk
import os
import glob
from PIL import ImageTk, Image

DATA_DIR = "../data"
DATASET_DIR = os.path.join(DATA_DIR, "dataset_condensed")
DIRECTION_FILE = os.path.join(DATA_DIR, "direction_condensed.txt")

jpgs = glob.glob(DATASET_DIR + "\\*.jpg")

directions = dict()


class Application(tk.Frame):
    def __init__(self, master=None):
        """Sets up state and keybinds."""
        super().__init__(master)
        self.master.focus_set()
        self.master.bind("<Left>", self.left)
        self.master.bind("<Right>", self.right)
        self.master.bind("<End>", self.goto_last)
        self.master.bind("<Ctrl-Left>", self.prev_unlabeled)
        self.master.bind("<Ctrl-Right>", self.next_unlabeled)
        self.master.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.master = master
        self.pack()
        self.current_index = 0
        self.load()
        self.create_widgets()

    def create_widgets(self):
        """Sets up interface."""
        self.label = tk.Label(self)
        self.label["text"] = ""
        self.label.pack(side="top")

        self.img = tk.Label(self)
        self.update()
        self.img["image"] = self.tk_image
        self.img.pack(side="top")

        self.btn_next = tk.Button(self)
        self.btn_next["text"] = "Next"
        self.btn_next["command"] = self.next
        self.btn_next.pack(side="right")

        self.btn_prev = tk.Button(self)
        self.btn_prev["text"] = "Previous"
        self.btn_prev["command"] = self.prev
        self.btn_prev.pack(side="left")

        self.btn_quit = tk.Button(self, text="QUIT", fg="red",
                              command=self.quit)
        self.btn_quit.pack(side="bottom")

    def quit(self):
        """Quits the program."""
        self.save()
        self.master.destroy()

    def on_closing(self):
        """Call this in the event of closing the window to save."""
        self.save()

    def load(self):
        """Loads the direction file."""
        with open(DIRECTION_FILE) as file:
            for line in file:
                line = line.strip()
                row = line.split(",")
                directions[row[0]] = int(row[1])


    def save(self):
        """Writes the directions to a file."""
        with open(DIRECTION_FILE, "w") as file:
            for key, value in directions.items():
                file.write(key + "," + str(value) + "\n")

    def update(self):
        """Updates the interface after changing image."""
        filename = jpgs[self.current_index]
        pil_image = Image.open(filename)
        pil_image = pil_image.resize((400, 300), Image.ANTIALIAS)
        self.tk_image = ImageTk.PhotoImage(pil_image)
        self.img["image"] = self.tk_image

        basename = os.path.basename(filename)
        if basename in directions.keys():
            self.label["text"] = "left" if directions[basename] == 0 else "right"
        else:
            self.label["text"] = ""

    def prev(self):
        """Go to the previous image."""
        if self.current_index > 0:
            self.current_index -= 1
            self.update()

    def next(self):
        """Go to the next image."""
        self.current_index += 1
        self.update()

    def left(self, e=None):
        """Label current as left and go the the next."""
        name = os.path.basename(jpgs[self.current_index])
        directions[name] = 0
        self.next()

    def right(self, e=None):
        """Label current as right and go to the next"""
        name = os.path.basename(jpgs[self.current_index])
        directions[name] = 1
        self.next()

    def goto_last(self, e=None):
        # TODO: fixme
        idx = len(directions.keys())
        self.current_index = idx
        self.update()

    def next_unlabeled(self, e=None):
        raise NotImplemented

    def prev_unlabeled(self, e=None):
        raise NotImplemented


root = tk.Tk()
root.title("Direction Labeler")
root.geometry("800x600")
app = Application(master=root)
app.mainloop()
