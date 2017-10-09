#!/usr/bin/env python

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec

matplotlib.rcParams.update({'font.size': 12})

alexnet_losses = [4.3052, 4.0601, 3.7327, 3.5282, 3.2629, 2.9588, 2.6607, 2.4724, 2.2321, 1.7490, 1.6729, 1.3894,
                  1.1409, 0.9931, 0.8733, 3.5727, 3.1244, 2.6956, 2.2900, 2.0428, 1.8441, 1.4933, 1.5324, 1.3044,
                  1.1815, 1.2532, 1.6017, 1.4790, 1.1573, 1.3542, 1.4608]
kaggle_losses = [9.7588, 7.2472, 5.8151, 6.7804, 5.6658, 6.8407, 5.3039, 4.7076, 5.1853, 7.2633, 6.0797, 5.3548, 5.5005,
                 5.1323, 4.9735, 5.0450, 5.3414, 5.2349, 3.5961, 3.9850, 4.6663, 5.5973]
resnet50_losses = [2.6536, 1.5876, 0.5563, 0.1151]


def plot_training_loss(ax, losses, num_epochs, title):
    losses = losses[0: num_epochs]
    ax.plot(losses, linewidth=1.5)
    ax.set_title(title)
    ax.grid(color='r', linestyle='-', linewidth=0.2)
    ax.set_xlabel('Epoch number')
    ax.set_ylabel('Training loss')


fig = plt.figure(figsize=(10, 3))
gs = gridspec.GridSpec(9, 95)
ax1 = fig.add_subplot(gs[0:9, 0: 25])
ax2 = fig.add_subplot(gs[0:9, 35: 60])
ax3 = fig.add_subplot(gs[0:9, 70: 95])

plot_training_loss(ax1, alexnet_losses, 20, 'AlexNet')
plot_training_loss(ax2, kaggle_losses, 20, 'Custom CNN')
plot_training_loss(ax3, resnet50_losses, 20, 'ResNet-50')

plt.savefig("geo_training_loss.png", dpi=100, bbox_inches='tight', pad_inches=0.1)
plt.show()
