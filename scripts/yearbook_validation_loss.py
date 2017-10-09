#!/usr/bin/env python

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec

matplotlib.rcParams.update({'font.size': 12})

alexnet_losses = [8.73687362747, 7.36693950888, 6.84348173288, 6.60311439409, 6.54062687163, 6.34897185067,
                  6.1137951687, 6.01936514274, 5.9083649431, 5.80315432222, 5.77081253743, 5.64803353963, 5.71730884408,
                  5.61728888002, 5.73128368936, 5.65741665003, 5.62966660012, 5.59073667399, 5.82192054302,
                  5.70113795169, 5.73008584548, 5.81692952685, 5.58215212617, 5.48472749052, 5.61389498902,
                  5.55639848273, 5.52445597924, 5.61609103613, 5.70453184268, 5.61469355161, 5.52725094829,
                  5.45078858056, 5.54182471551, 5.54442004392, 5.66739868237, 5.57296865642, 5.49151527251,
                  5.60211619086, 5.54921141944, 5.69954082651, 5.47634258335, 5.53523657417, 5.61149930126,
                  5.51766819724, 5.30425234578, 5.44519864244, 5.53823118387, 5.36933519665, 5.36594130565,
                  5.57276901577, 5.60411259732, 5.52525454182, 5.57636254741, 5.64983030545, 5.50149730485,
                  5.61149930126, 5.3375923338, 5.42064284288, 5.52884807347, 5.65422239968, 5.50109802356,
                  5.66400479138, 5.55739668597, 5.32341784787, 5.54302255939, 5.57177081254, 5.61030145738,
                  5.80854461968, 5.59413056498, 5.38390896387, 5.54442004392, 5.45937312837, 5.62467558395,
                  5.47155120783, 5.57895787582, 5.41185865442, 5.40786584149, 5.49890197644, 5.63845078858,
                  5.51926532242, 5.58215212617, 5.36194849271, 5.51267718107, 5.56638051507, 5.59952086245,
                  5.5486124975, 5.43761229786, 5.39968057497, 5.59233379916, 5.54941106009, 5.44100618886,
                  5.61788780196, 5.55300459173, 5.61449391096, 5.58794170493, 5.43661409463, 5.57037332801,
                  5.54701537233, 5.44619684568, 5.60790576961, 5.60790576961]
vgg16_losses = [9.33779197445, 7.74925134757, 7.48233180276, 6.92433619485, 6.28568576562, 6.43022559393, 7.13535635855,
                6.92832900779, 7.24376122979, 7.12936713915, 6.94090636854, 6.85146735875]
vgg19_losses = [8.45478139349, 6.66480335396, 6.42543421841, 6.27670193651, 6.15591934518, 6.93531643043, 6.03274106608,
                6.83270113795]
densenet169_losses = [10.1495308445, 7.34977041326, 5.96526252745, 5.58195248553, 5.91495308445, 5.27031343582,
                      5.16470353364, 5.02974645638, 5.01776801757, 5.02635256538, 5.0694749451]
densenet121_losses = [11.0764623677, 6.96985426233, 6.76322619285, 6.55400279497, 6.24515871431, 6.06568177281]
resnet50_losses = [10.0473148333, 5.7825913356, 5.84727490517]


def plot_validation_loss(ax, losses, num_epochs, title):
    losses = losses[0: num_epochs]
    if title == 'ResNet-50':
        ax.plot(x=[1,2, 3, 6, 11], y=losses, linewidth=1.5)
        ax.set_xlim([0, 12])
        ax.set_ylim([0, 12])
    else:
        ax.plot(losses, linewidth=1.5)
    ax.set_title(title)
    ax.grid(color='r', linestyle='-', linewidth=0.2)
    ax.set_xlabel('Epoch number')
    ax.set_ylabel('Validation loss (L1)')


fig = plt.figure(figsize=(12, 7))
gs = gridspec.GridSpec(22, 95)
ax1 = fig.add_subplot(gs[0:9, 0: 25])
ax2 = fig.add_subplot(gs[0:9, 35: 60])
ax3 = fig.add_subplot(gs[0:9, 70: 95])

ax4 = fig.add_subplot(gs[13:22, 0: 25])
ax5 = fig.add_subplot(gs[13:22, 35: 60])
ax6 = fig.add_subplot(gs[13:22, 70: 95])

plot_validation_loss(ax1, alexnet_losses, 20, 'AlexNet')
plot_validation_loss(ax2, vgg16_losses, 20, 'VGG16')
plot_validation_loss(ax3, vgg19_losses, 20, 'VGG19')
plot_validation_loss(ax4, resnet50_losses, 20, 'ResNet-50')
plot_validation_loss(ax5, densenet169_losses, 20, 'DenseNet-169')
plot_validation_loss(ax6, densenet121_losses, 20, 'DenseNet-121')

plt.savefig("yearbook_validation_loss.png", dpi=100, bbox_inches='tight', pad_inches=0.1)
plt.show()
