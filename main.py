import math
import os.path
import sys
import time

import numpy as np
from PIL import Image
from PyQt5.QtGui import QPixmap, QImage, QDoubleValidator
from PyQt5.QtWidgets import QMainWindow, QGraphicsView, QLabel, QApplication, QPushButton, QFileDialog
from PyQt5.uic import loadUi
from matplotlib import pyplot as plt
from pybrain3 import FeedForwardNetwork, RecurrentNetwork, BiasUnit, FullConnection, LinearLayer, SigmoidLayer
from pybrain3.supervised import BackpropTrainer
from pybrain3.tools.shortcuts import buildNetwork
from pybrain3.datasets.supervised import SupervisedDataSet
from pybrain3 import FeedForwardNetwork
from PyQt5 import QtCore as qtc
import pickle


__VERSION__ = "2.0.1"

def createDataSet(rgb_src, gray_src):
    print("---- createDataSet ----")

    dataset = SupervisedDataSet(100, 300)

    img = Image.open(rgb_src)
    rgb_color = img.load()
    print(img)

    img = Image.open(gray_src)
    gray_color = img.load()
    print(img)

    h, w = img.size
    print("height", h)
    print("width", w)

    input_data_blocks = np.array([])
    target_data_blocks = np.array([])
    for y in range(0, h, 10):
        for x in range(0, w, 10):
            # print(f'input_data_blocks = {input_data_blocks} \n\n target_data_blocks =  {target_data_blocks}')
            if (input_data_blocks.size == 100) and (target_data_blocks.size == 300):
                dataset.addSample(input_data_blocks, target_data_blocks)

            input_data_blocks = np.array([])
            target_data_blocks = np.array([])

            for y1 in range(10):
                for x1 in range(10):
                    if not ((x + x1 >= w) or (y + y1 >= h)):
                        # print(f'x = {x}\nx1 = {x1}\ny = {y}\ny1 = {y1}')
                        red, green, blue = rgb_color[y + y1, x + x1]
                        gray = gray_color[y + y1, x + x1] / 256
                        input_data_blocks = np.append(input_data_blocks, [gray])
                        target_data_blocks = np.append(target_data_blocks, [red / 256, green / 256, blue / 256])

    print(dataset)
    return dataset


def arraySum(array):
    _sum = ()
    for value in array:
        _sum += value
    return _sum


class MainWindow(QMainWindow):  # главное окно
    def __init__(self, parent=None):
        super().__init__(parent)
        # self.setupUi()

        self.net_name = None
        loadUi("ui/color_restavration.ui", self)

        self.graphic_view = self.findChild(QLabel, "ImageBeforeView")
        self.graphic_view2 = self.findChild(QLabel, "ImageAfterView")
        print(f'graphic_view = {self.graphic_view}')

        self.loadImage.clicked.connect(self.loadInputImg)
        self.startImageProcessing.clicked.connect(self.startNet)
        self.startLearnButton.clicked.connect(self.runTrainig)

    def runTrainig(self):
        print("---- runTrainig ----")
        self.create_new_net = CreateNewNetWindow()
        self.create_new_net.submitClicked.connect(self.onNewNetNameConfirm)
        self.create_new_net.show()

    def onNewNetNameConfirm(self, name, learning_rate,epochs):
        if not name:
            self.net_name = "color_restovration_neuro_net_model.txt"
        else:
            self.net_name = name + ".txt"

        print("name:", name)
        print("learning_rate:",learning_rate)
        print("learning epochs count:",epochs)

        startTraining(self.image_rgb_file_name, self.input_image_file_name, epochs, self.net_name, learning_rate)

    def startNet(self):
        print("---- startNet ----")

        if not self.input_image_file_name:
            return

        newFileName = self.input_image_file_name[
                      0:self.input_image_file_name.index(".")] + "_processed" + self.input_image_file_name[
                                                                                self.input_image_file_name.index("."):]
        print(f'newFileName={newFileName}')
        if not self.net_name:
            self.net_name = self.selectNet()
        elif not self.net_name.index("."):
            self.net_name = "neuralNetworks/v" + __VERSION__ + self.net_name + ".txt"


        runNet(self.input_image_file_name, newFileName, self.net_name)

        image_qt = QImage(newFileName)

        self.graphic_view2.setPixmap(QPixmap.fromImage(image_qt))

    def selectNet(self):
        net_src, _ = QFileDialog.getOpenFileName(
            self, "Open file", ".", "Text files (*.txt)"
        )
        if not net_src:
            return ''
        return net_src

        pass

    def loadInputImg(self):
        self.input_image_file_name, _ = QFileDialog.getOpenFileName(
            self, "Open file", ".", "Image Files (*.png *.jpg *.bmp)"
        )
        if not self.input_image_file_name:
            return
        img = Image.open(self.input_image_file_name)
        if img.mode != 'L':
            img = img.convert('L')
            newFileName = self.input_image_file_name[
                          0:self.input_image_file_name.index(".")] + "_gray" + self.input_image_file_name[
                                                                               self.input_image_file_name.index("."):]
            img.save(newFileName)
            self.image_rgb_file_name = self.input_image_file_name
            self.input_image_file_name = newFileName

        image_qt = QImage(self.input_image_file_name)

        self.graphic_view.setPixmap(QPixmap.fromImage(image_qt))
        self.graphic_view2.setPixmap(QPixmap.fromImage(image_qt))

    def setupUi(self):
        self.setWindowTitle("Hello, world")  # заголовок окна
        # self.move(300, 300) # положение окна
        # self.resize(200, 200) # размер окна


class CreateNewNetWindow(QMainWindow):  # окно добавления новой сети
    submitClicked = qtc.pyqtSignal(str, float,int)

    def __init__(self, parent=None):
        super().__init__(parent)
        # self.setupUi()

        loadUi("ui/create_new_net_window.ui", self)
        self.create_button.clicked.connect(self.confirm)

    def confirm(self):
        self.submitClicked.emit(self.nameTextBox.text(),self.learning_rate.value(),self.epochs_count.value())
        self.close()


def run():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())

    # runNet("resources/images/Les_ottenki_serogo.jpg","resources/images/Les_ottenki_serogo_preobrazovano.jpg")


def runNet(img_src: str, output_image_src: str, net_name: str):
    print('---- runNet ----')

    net = loadNetFromFile(net_name)
    # print(net.activate([n / 100 for n in range(100)]))

    img = Image.open(img_src)
    print(f'image mode={img.mode}')
    h, w = img.size
    data = prepareData(img)

    # for block in data:
    #     print("----block:", len(block))

    newImage = Image.new("RGB", size=img.size)

    loadingIteration = 1

    x = 0
    y = 0
    for block in data:
        processed_block = net.activate(block)

        printProgressBar(loadingIteration, len(data), prefix='Progress:', suffix='Complete', length=50)
        loadingIteration += 1

        elem = 0
        for y1 in range(10):
            for x1 in range(10):
                if x + x1 < w and y + y1 < h:
                    r, g, b = processed_block[elem * 3], processed_block[elem * 3 + 1], processed_block[elem * 3 + 2]
                    elem += 1;
                    newImage.putpixel((y + y1, x + x1), (int(r * 256), int(g * 256), int(b * 256)))

        x += 10
        if x + 10 > w:
            y += 10
            x = 0

    print(f'output_image_src={output_image_src}')
    newImage.save(output_image_src)


def prepareData(img: Image):
    image = img.load()
    h, w = img.size

    data = []
    data_block = []
    for y in range(0, h, 10):
        for x in range(0, w, 10):
            # print("--data_block",data_block,len(data_block))
            # time.sleep(0.1)
            if len(data_block) == 100:
                data.append(data_block)
                # print(data)
            data_block = []
            for y1 in range(10):
                for x1 in range(10):
                    if not ((x + x1 >= w) or (y + y1 >= h)):
                        gray = image[y + y1, x + x1] / 256
                        data_block.append(gray)

    return data


def buildMyNetwork():
    nn = FeedForwardNetwork()

    bias = BiasUnit()
    inLayer = LinearLayer(100)
    hidden_layer1 = SigmoidLayer(200)
    hidden_layer2 = SigmoidLayer(200)
    outLayer = LinearLayer(300)

    bias.name = 'bias'
    inLayer.name = 'in'
    hidden_layer1.name = 'hidden1'
    hidden_layer2.name = 'hidden2'
    outLayer.name = 'out'

    nn.addInputModule(inLayer)
    nn.addModule(bias)
    nn.addModule(hidden_layer1)
    nn.addModule(hidden_layer2)
    nn.addOutputModule(outLayer)

    nn.addConnection(FullConnection(inLayer, hidden_layer1))
    nn.addConnection(FullConnection(hidden_layer1, hidden_layer2))
    nn.addConnection(FullConnection(hidden_layer2, outLayer))

    bias_to_hidden1 = FullConnection(bias, hidden_layer1)
    bias_to_hidden2 = FullConnection(bias, hidden_layer2)
    bias_to_out = FullConnection(bias, outLayer)
    nn.addConnection(bias_to_hidden1)
    nn.addConnection(bias_to_hidden2)
    nn.addConnection(bias_to_out)

    nn.sortModules()

    return nn


def startTraining(rgb: str, gray: str, epochs: int, net_name: str, learning_rate: float):
    print("---- startTraining ----")
    # net = buildNetwork(100, 3, 300)
    net = buildMyNetwork()
    print(net)
    ds = createDataSet(rgb, gray)

    # y = net.activate()
    # print("Y= ", y)
    # printNetParameters(net)

    tranier = BackpropTrainer(module=net, dataset=ds, learningrate=learning_rate)

    # trnerr, valerr = tranier.trainUntilConvergence(dataset=ds, maxEpochs=10, continueEpochs=5)

    trnerr = np.array([]);

    for i in range(epochs):
        err = tranier.train()
        trnerr = np.append(trnerr, [err])

        print("--epoch number=", i, "\ttraining error=", err)
    print(trnerr)

    # График ошибки проверки и ошибки обучения
    plt.plot(trnerr, 'b')
    plt.show()

    # Сохранение сети в файл
    saveNetToFile(net_name, net)


def saveNetToFile(fileName, net):
    if not os.path.exists("neuralNetworks/v" + __VERSION__):
        os.makedirs("neuralNetworks/v" + __VERSION__)

    fileObject = open("neuralNetworks/v" + __VERSION__ + "/" + fileName, 'wb')
    pickle.dump(net, fileObject)
    fileObject.close()
    print("Successfully saved network to ", fileName)


def loadNetFromFile(fileName) -> FeedForwardNetwork | RecurrentNetwork:
    fileObject = open(fileName, 'rb')
    return pickle.load(fileObject)


# Print iterations progress
def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', printEnd="\r"):
    # print(f'iteration={iteration} total={total}')

    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end="")
    # Print New Line on Complete
    if iteration == total:
        print()


def printNetParameters(net):
    for mod in net.modules:
        print("Module:", mod.name)
        if mod.paramdim > 0:
            print("--parameters", mod.params)
        for conn in net.connections[mod]:
            print("-connection to", conn.outmod.name)
            if conn.paramdim > 0:
                print("-parameters", conn.params)

        if hasattr(net, "recurrentConns"):
            print("Reccurent connections")
            for conn in net.recurrentConns:
                print("-", conn.inmod.name, " to", conn.outmod.name)
                if conn.paramdim > 0:
                    print("-parameters", conn.params)


if __name__ == '__main__':
    run()
