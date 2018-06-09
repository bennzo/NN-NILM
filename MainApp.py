#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *

import MainAppDesign
import data
import network
import utilities



class MainApp(QMainWindow, MainAppDesign.Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainApp, self).__init__(parent)

        # ---------- Set Layout ----------- #
        self.setupUi(parent)

        # ---------- Resources ----------- #
        self.dataPath = None
        self.labelPath = None

        self.app1_chk = False
        self.app2_chk = False
        self.app3_chk = False
        self.app4_chk = False
        self.app5_chk = False
        self.app_combs = []

        self.signal = None
        self.signal_file = None

        # ---------- Control ------------- #
        self.btn_loadData.clicked.connect(self.load_data)

        self.cbx_app1.stateChanged.connect(lambda: self.checked_app_box(1))
        self.cbx_app2.stateChanged.connect(lambda: self.checked_app_box(2))
        self.cbx_app3.stateChanged.connect(lambda: self.checked_app_box(3))
        self.cbx_app4.stateChanged.connect(lambda: self.checked_app_box(4))
        self.cbx_app5.stateChanged.connect(lambda: self.checked_app_box(5))
        self.btn_addComb.clicked.connect(self.add_comb)
        self.btn_clearComb.clicked.connect(self.clear_comb)
        self.btn_plotComb.clicked.connect(self.plot_signal)

        self.btn_trainAll.clicked.connect(lambda: self.train_net(comb=False))
        self.btn_train.clicked.connect(lambda: self.train_net(comb=True))

        self.btn_loadSignal.clicked.connect(self.load_signal)
        self.btn_disagg.clicked.connect(self.disaggregate_signal)

    def load_data(self):
        self.dataPath = str(QFileDialog.getExistingDirectory(self, "Select Directory")) + '//'

    def load_signal(self):
        signal_path = QFileDialog.getOpenFileName(self, "Select File")[0]
        self.signal = pd.read_csv(signal_path, header=None).values.flatten()

    def disaggregate_signal(self):
        network.disaggregate(self.signal)

    def checked_app_box(self,n):
        if (n == 1): self.app1_chk = not self.app1_chk
        if (n == 2): self.app2_chk = not self.app2_chk
        if (n == 3): self.app3_chk = not self.app3_chk
        if (n == 4): self.app4_chk = not self.app4_chk
        if (n == 5): self.app5_chk = not self.app5_chk

    def add_comb(self):
        comb_rep = np.array([self.app1_chk,self.app2_chk,self.app3_chk,self.app4_chk,self.app5_chk]).astype(int)
        comb = '0b' + str(int(self.app1_chk)) + str(int(self.app2_chk)) +\
                      str(int(self.app3_chk)) + str(int(self.app4_chk)) + str(int(self.app5_chk))

        if (comb not in self.app_combs):
            self.app_combs.append(comb)
            self.lst_sigCombs.addItem(str(comb_rep))

    def clear_comb(self):
        self.lst_sigCombs.clear()
        self.app_combs.clear()

    def plot_signal(self):
        if self.signal_file is None:
            self.signal_file = pd.read_csv(self.dataPath + 'signal_sum_val.txt', header=None).values.flatten()

        comb = '0b' + str(int(self.app1_chk)) + str(int(self.app2_chk)) +\
                      str(int(self.app3_chk)) + str(int(self.app4_chk)) + str(int(self.app5_chk))

        sig_len = self.signal_file.size//(2**5)
        win = 128*2
        utilities.plot_signal_gui(self.signal_file[int(comb,2)*sig_len:int(comb,2)*sig_len+win])

    def train_net(self, comb):
        self.txt_resultsText.clear()
        if (self.dataPath is not None):
            if (comb and len(self.app_combs) > 0):
                data.train_comb = self.app_combs
                network.train(self.dataPath, lambda path: data.data_init_comb(path, train_comb=self.app_combs, test_comb=self.app_combs))
            if (not comb):
                network.train(self.dataPath, data.data_init_measured)

# Logger class
# Redirects stdout to the result text frame in the gui
class OutLog:
    def __init__(self, edit, out=None, color=None):
        # edit = QTextEdit
        # out = alternate stream (can be the original sys.stdout)
        # color = alternate color (i.e. color stderr a different color)
        self.edit = edit
        self.out = None
        self.color = color

    def write(self, m):
        if self.color:
            tc = self.edit.textColor()
            self.edit.setTextColor(self.color)

        self.edit.moveCursor(QTextCursor.End)
        self.edit.insertPlainText( m )

        if self.color:
            self.edit.setTextColor(tc)

        if self.out:
            self.out.write(m)


def main():
    app = QApplication(sys.argv)
    form = QMainWindow()
    ui = MainApp(parent=form)
    form.show()

    sys.stdout = OutLog(ui.txt_resultsText)

    app.exec_()

if __name__ == "__main__":
    main()