#!/usr/bin/env python

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec

matplotlib.rcParams.update({'font.size': 12})

alexnet_losses = [2.6392, 1.4771, 1.0919, 0.7296, 0.4928, 0.2740, 0.1611, 0.0994, 0.1264, 0.1238, 0.1119, 0.1065,
                  0.0387,
                  0.0272, 0.0243, 0.0450, 0.0777, 0.0288, 0.0264, 0.0138, 0.0328, 0.0087, 0.0289, 0.0341, 0.0121,
                  0.0043,
                  0.0024, 0.0116, 0.0198, 0.0153, 0.0060, 0.0459, 0.0045, 0.0043, 0.0031, 0.0052, 0.0049, 0.0026,
                  0.0115,
                  0.0021, 0.0015, 0.0235, 0.0054, 0.0030, 0.0157, 0.0023, 0.0031, 0.0039, 7.1618e-04, 0.0049]
vgg16_losses = [2.2667, 1.1647, 0.7358, 0.4410, 0.2405, 0.1581, 0.0830, 0.0456, 0.0369, 0.0288, 0.0135, 0.0052]
vgg19_losses = [2.2363, 1.4173, 0.5918, 0.4204, 0.2709, 0.2301, 0.1267, 0.1162, 0.0491, 0.0215]
densenet169_losses = [1.8676, 0.7383, 0.2053, 0.0701, 0.0250, 0.0173, 0.0068, 0.0043, 0.0033, 0.0028, 0.0024]
densenet121_losses = [1.5101, 0.8548, 0.2705, 0.1045, 0.0715, 0.0246]
resnet50_losses = [0.8468, 0.1705, 0.0482, 0.0101, 0.0049, 0.0021, 0.0013, 0.0011, 8.9024e-04, 7.7238e-04, 6.8226e-04,
                   6.1079e-04]
resnet152_losses = []  # immature model


def plot_training_loss(ax, losses, num_epochs, title):
    losses = losses[0: num_epochs]
    ax.plot(losses, linewidth=1.5)
    ax.set_title(title)
    ax.grid(color='r', linestyle='-', linewidth=0.2)
    ax.set_xlabel('Epoch number')
    ax.set_ylabel('Training loss')


fig = plt.figure(figsize=(12, 7))
gs = gridspec.GridSpec(22, 95)
ax1 = fig.add_subplot(gs[0:9, 0: 25])
ax2 = fig.add_subplot(gs[0:9, 35: 60])
ax3 = fig.add_subplot(gs[0:9, 70: 95])

ax4 = fig.add_subplot(gs[13:22, 0: 25])
ax5 = fig.add_subplot(gs[13:22, 35: 60])
ax6 = fig.add_subplot(gs[13:22, 70: 95])

plot_training_loss(ax1, alexnet_losses, 20, 'AlexNet')
plot_training_loss(ax2, vgg16_losses, 20, 'VGG16')
plot_training_loss(ax3, vgg19_losses, 20, 'VGG19')
plot_training_loss(ax4, resnet50_losses, 20, 'ResNet-50')
plot_training_loss(ax5, densenet169_losses, 20, 'DenseNet-169')
plot_training_loss(ax6, densenet121_losses, 20, 'DenseNet-121')

plt.savefig("yearbook_training_loss.png", dpi=100, bbox_inches='tight', pad_inches=0.1)
plt.show()
