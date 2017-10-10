import sys
import numpy as np


def get_labels(file_name):
    r = []
    r += [int(n.strip().split('\t')[1]) for n in open(file_name, 'r')]
    return np.array(r)


def get_l1_distance(labels1, labels2):
    print(labels1.shape)
    num = len(labels1)
    sum = float(np.sum(abs(labels1 - labels2)))
    return sum / num


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise Exception('Please give two files as arguments!')

    file1 = sys.argv[1]
    file2 = sys.argv[2]

    labels1 = get_labels(file1)
    labels2 = get_labels(file2)

    print('L1 distance: ' + str(get_l1_distance(labels1, labels2)))
