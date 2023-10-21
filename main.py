import math
import sys

import numpy as np
from PIL import Image
from PyQt5.QtWidgets import QMainWindow, QLabel, QApplication
from PyQt5.uic import loadUi
from matplotlib import pyplot as plt
from pybrain3 import FeedForwardNetwork, RecurrentNetwork
from pybrain3.supervised import BackpropTrainer
from pybrain3.tools.shortcuts import buildNetwork
from pybrain3.datasets.supervised import SupervisedDataSet
import pickle


def createDataSet(rgb_src, gray_src):
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
            if (input_data_blocks.size > 0) and (target_data_blocks.size > 0):
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


class MainWindow(QMainWindow): # главное окно
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi()
    def setupUi(self):
        self.setWindowTitle("Hello, world") # заголовок окна
        # self.move(300, 300) # положение окна
        # self.resize(200, 200) # размер окна

        loadUi("color_restavration.ui",self)

def run():
    # app = QApplication(sys.argv)
    # win = MainWindow()
    # win.show()
    # sys.exit(app.exec_())

    net = loadNetFromFile('color_restovration_neuro_net_model.txt')
    # print(net.activate([n / 100 for n in range(100)]))

    img = Image.open("resources/images/Les_ottenki_serogo.jpg")
    h, w = img.size
    data = prepareData(img)

    newImage = Image.new("RGB",size=img.size)

    x = 0
    y = 0
    for block in data:
        processed_block = net.activate(block);
        print(block)
        print('------')
        print(processed_block)

        elem = 0
        for y1 in range(10):
            for x1 in range(10):
                if x + x1 < w and y + y1 < h:
                    r, g, b = processed_block[elem * 3], processed_block[elem * 3 + 1], processed_block[elem * 3 + 2]
                    elem += 1;
                    newImage.putpixel((y + y1, x + x1), (int(r * 256), int(g * 256), int(b * 256)))


        x += 10
        y += 10

    newImage.save("resources/images/Les_ottenki_serogo_preobrazovano.jpg")


def prepareData(img : Image):
    image = img.load()
    h, w = img.size

    data = []
    data_block = []
    for y in range(0, h, 10):
        for x in range(0, w, 10):
            if len(data_block) > 0:
                data.append(data_block)
                # print(data)
                data_block = []
            for y1 in range(10):
                for x1 in range(10):
                    if not ((x + x1 >= w) or (y + y1 >= h)):
                        gray = image[y + y1, x + x1] / 256
                        data_block.append(gray)

    return data

def startTraining():
    net = buildNetwork(100, 1, 300)
    ds = createDataSet("resources/images/Les.jpg", "resources/images/Les_ottenki_serogo.jpg")

    # y = net.activate()
    # print("Y= ", y)
    print(net)
    # printNetParameters(net)

    tranier = BackpropTrainer(net, ds)

    # trnerr, valerr = tranier.trainUntilConvergence(dataset=ds, maxEpochs=10, continueEpochs=5)

    trnerr = np.array([]);

    for i in range(10):
        err = tranier.train()
        trnerr = np.append(trnerr, [err])

        print(err)
    print(trnerr)

    # График ошибки проверки и ошибки обучения
    plt.plot(trnerr, 'b')
    plt.show()

    # Сохранение сети в файл
    saveNetToFile('color_restovration_neuro_net_model.txt', net)

def saveNetToFile(fileName, net):
    fileObject = open(fileName, 'wb')
    pickle.dump(net, fileObject)
    fileObject.close()


def loadNetFromFile(fileName) -> FeedForwardNetwork | RecurrentNetwork:
    fileObject = open(fileName, 'rb')
    return pickle.load(fileObject)


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
