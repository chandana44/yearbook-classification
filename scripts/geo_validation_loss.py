#!/usr/bin/env python

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec

matplotlib.rcParams.update({'font.size': 12})

alexnet_losses = [385.370440136, 374.845486651, 370.710639865, 371.867976693, 367.448298284, 363.357791484,
                  367.205507758, 357.997757285, 362.087212748, 364.081771535, 365.569697387, 362.21037668,
                  364.957544806, 373.155298103, 371.87867172, 369.552660091, 368.782989524, 372.997127604,
                  378.204474752, 380.919032425, 383.014202436, 393.203762065, 399.584390402, 394.795908135,
                  390.026636351, 399.694337453, 401.077588217, 411.932175805, 405.446062186, 406.362433351,
                  404.817494341]
kaggle_losses = [366.084455055, 345.80642691, 345.397004147, 339.134056109, 339.287110352, 338.270762341, 350.28376408,
                 332.342284059, 327.926905394, 329.228776525, 326.28368264, 323.620591825, 321.057782845, 316.665648582,
                 315.176115303, 318.403077017, 316.049906288, 316.685807061, 313.256421195, 309.788554445,
                 309.203816623]
resnet50_losses = [318.602571992, 305.557804497, 305.251880709, 306.45039682]


def plot_validation_loss(ax, losses, num_epochs, title):
    losses = losses[0: num_epochs]
    ax.plot(losses, linewidth=1.5)
    ax.set_title(title)
    ax.grid(color='r', linestyle='-', linewidth=0.2)
    ax.set_xlabel('Epoch number')
    ax.set_ylabel('Validation loss (L1)')


fig = plt.figure(figsize=(10, 3))
gs = gridspec.GridSpec(9, 95)
ax1 = fig.add_subplot(gs[0:9, 0: 25])
ax2 = fig.add_subplot(gs[0:9, 35: 60])
ax3 = fig.add_subplot(gs[0:9, 70: 95])

plot_validation_loss(ax1, alexnet_losses, 20, 'AlexNet')
plot_validation_loss(ax2, kaggle_losses, 20, 'Custom CNN')
plot_validation_loss(ax3, resnet50_losses, 20, 'ResNet-50')

plt.savefig("geo_validation_loss.png", dpi=100, bbox_inches='tight', pad_inches=0.1)
plt.show()
