import matplotlib.pyplot as plt
import numpy as np
import torch


def display(data_list, num_epoch, labels, title):
    plt_x = np.linspace(1, num_epoch, num_epoch)
    for i in range(2):
        for j in range(2):
            ax = plt.subplot(221 + i * 2 + j)
            d_list = data_list[i * 2 + j]
            ax.set_title(labels[i * 2 + j])
            plt.plot(plt_x, d_list, label=labels[i * 2 + j])

    plt.show()


def display_one(data_list, num_epoch, labels, title):
    plt_x = np.linspace(1, num_epoch, num_epoch)
    l1, = plt.plot(plt_x, np.asarray(data_list[0]) / 2.0, label=labels[0])
    l2, = plt.plot(plt_x, np.asarray(data_list[1]) / 30.0, label=labels[1])
    l3, = plt.plot(plt_x, np.asarray(data_list[2]) / 30.0, label=labels[2])
    l4, = plt.plot(plt_x, np.asarray(data_list[3]) / 3000.0, label=labels[3])

    plt.title(title)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(handles=[l1, l2, l3, l4], labels=labels, loc='best')

    plt.show()
