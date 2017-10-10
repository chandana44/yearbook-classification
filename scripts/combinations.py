import itertools

alex_p = 'alexnet:checkpoint52-44-weights.h5'
vgg16_p = 'vgg16:checkpoint48-4.h5'
vgg19_p = 'vgg19:checkpoint49-4-7.h5'
dense169_p = 'densenet169:checkpoint13-0-1-2-3-4-5-6-7-8.h5'
dense121_p = 'densenet121:checkpoint50-0-1-2-3-4-5.h5'
resnet50_p = 'resnet50:resnet50checkpoint18-5.h5'
resnet152_p = 'resnet152:resnet152checkpoint3-0-1.h5'

arr = [alex_p, vgg16_p, vgg19_p, dense169_p, dense121_p, resnet50_p, resnet152_p]


def func(subset):
    if len(subset) <= 0:
        return
    return ','.join(subset)


all_combinations = []
for L in range(0, len(arr) + 1):
    for subset in itertools.combinations(arr, L):
        combination = func(subset)
        if combination is not None:
            all_combinations.append(combination)
        print(combination)
        print('-'*40)

print('#'.join(all_combinations))
