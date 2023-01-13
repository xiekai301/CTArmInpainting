from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QIcon,QPixmap,QFont
from PyQt5.QtCore import Qt

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(1200, 660)
        self.pushButton = QtWidgets.QPushButton(Form)
        self.pushButton.setGeometry(QtCore.QRect(10, 40, 97, 27))
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(Form)
        self.pushButton_2.setGeometry(QtCore.QRect(130, 10, 97, 27))
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_3 = QtWidgets.QPushButton(Form)
        self.pushButton_3.setGeometry(QtCore.QRect(250, 10, 97, 27))
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_4 = QtWidgets.QPushButton(Form)
        self.pushButton_4.setGeometry(QtCore.QRect(10, 10, 97, 27))
        self.pushButton_4.setObjectName("pushButton_4")
        self.pushButton_5 = QtWidgets.QPushButton(Form)
        self.pushButton_5.setGeometry(QtCore.QRect(560, 360, 81, 27))
        self.pushButton_5.setObjectName("pushButton_5")
        self.pushButton_6 = QtWidgets.QPushButton(Form)
        self.pushButton_6.setGeometry(QtCore.QRect(490, 10, 97, 27))
        self.pushButton_6.setObjectName("pushButton_6")
        # self.pushButton_7 = QtWidgets.QPushButton(Form)
        # self.pushButton_7.setGeometry(QtCore.QRect(490, 10, 97, 27))
        # self.pushButton_7.setObjectName("pushButton_7")
        self.pushButton_8 = QtWidgets.QPushButton(Form)
        self.pushButton_8.setGeometry(QtCore.QRect(370, 10, 97, 27))
        self.pushButton_8.setObjectName("pushButton_8")
        self.pushButton_9 = QtWidgets.QPushButton(Form)
        self.pushButton_9.setGeometry(QtCore.QRect(880, 83, 105, 27))
        self.pushButton_9.setObjectName("pushButton_8")

        self.splider = QtWidgets.QSlider(Qt.Horizontal, Form)  # TODO: Qt.Horizontal
        self.splider.setMinimum(5)  # 最小值        24
        self.splider.setMaximum(30)  # 最大值        25
        self.splider.setSingleStep(2)  # 步长
        self.splider.setValue(10)
        self.splider.setGeometry(QtCore.QRect(130, 40, 97, 27))
        self.splider.setObjectName("pushButton_9")

        self.label = QtWidgets.QLabel(Form)
        self.label.setGeometry(QtCore.QRect(250, 40, 97, 27))
        self.label.setObjectName("pushButton_10")

        self.graphicsView = QtWidgets.QGraphicsView(Form)
        self.graphicsView.setGeometry(QtCore.QRect(20, 120, 512, 512))
        self.graphicsView.setObjectName("graphicsView")
        self.graphicsView_2 = QtWidgets.QGraphicsView(Form)
        self.graphicsView_2.setGeometry(QtCore.QRect(660, 120, 512, 512))
        self.graphicsView_2.setObjectName("graphicsView_2")

        self.saveImg = QtWidgets.QPushButton(Form)
        self.saveImg.setGeometry(QtCore.QRect(610, 10, 97, 27))
        self.saveImg.setObjectName("saveImg")

        # self.arrangement = QtWidgets.QPushButton(Form)
        # self.arrangement.setGeometry(QtCore.QRect(610, 40, 97, 27))
        # self.arrangement.setObjectName("arrangement")

        self.retranslateUi(Form)
        self.pushButton.clicked.connect(Form.open)
        self.pushButton_2.clicked.connect(Form.mask_mode)
        self.pushButton_3.clicked.connect(Form.sketch_mode)
        self.pushButton_4.clicked.connect(Form.open_model)
        self.pushButton_5.clicked.connect(Form.complete)
        self.pushButton_6.clicked.connect(Form.undo)
        # self.pushButton_7.clicked.connect(Form.color_change_mode)
        self.pushButton_8.clicked.connect(Form.clear)
        self.pushButton_9.clicked.connect(Form.change_ori_res)
        self.splider.valueChanged.connect(Form.valChange)

        self.saveImg.clicked.connect(Form.save_img)

        # self.arrangement.clicked.connect(Form.arrange)



        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Truncated Area Inpainting of CT images"))
        self.pushButton.setText(_translate("Form", "Open Image"))
        self.pushButton_2.setText(_translate("Form", "Mask"))
        self.pushButton_3.setText(_translate("Form", "Sketches"))
        self.pushButton_4.setText(_translate("Form", "Open Model"))
        self.pushButton_5.setText(_translate("Form", "Complete"))
        self.pushButton_6.setText(_translate("Form", "Undo"))
        # self.pushButton_7.setText(_translate("Form", "Palette"))
        self.pushButton_8.setText(_translate("Form", "Clear"))
        self.pushButton_9.setText(_translate("Form", "Origin/Result"))

        self.saveImg.setText(_translate("Form", "Save Img"))

        # self.arrangement.setText(_translate("Form", "Arrange"))
        self.label.setNum(10)


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Form = QtWidgets.QWidget()
    ui = Ui_Form()
    ui.setupUi(Form)
    Form.show()
    sys.exit(app.exec_())
