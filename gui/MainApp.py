#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tkinter as tk

from MainAppController import MainAppController

def main():
    controller = MainAppController()

    root = tk.Tk()
    root.title('NILM NN')

    controller.init_view(root)

if __name__ == "__main__":
    main()