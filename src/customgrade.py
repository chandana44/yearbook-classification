from __future__ import print_function

import os

os.environ['THEANO_FLAGS'] = "device=cuda0"

from argparse import ArgumentParser
from model import *
from run import *
from util import *

SRC_PATH = path.dirname(path.abspath(__file__))
DATA_PATH = path.join(SRC_PATH, '..', 'data')
YEARBOOK_PATH = path.join(DATA_PATH, 'yearbook')
YEARBOOK_VALID_PATH = path.join(YEARBOOK_PATH, 'valid')
YEARBOOK_TEST_PATH = path.join(YEARBOOK_PATH, 'test')
YEARBOOK_TEST_LABEL_PATH = path.join(SRC_PATH, '..', 'output', 'yearbook_test_label.txt')
STREETVIEW_PATH = path.join(DATA_PATH, 'geo')
STREETVIEW_VALID_PATH = path.join(STREETVIEW_PATH, 'valid')
STREETVIEW_TEST_PATH = path.join(STREETVIEW_PATH, 'test')
STREETVIEW_TEST_LABEL_PATH = path.join(SRC_PATH, '..', 'output', 'geo_test_label.txt')

CHECKPOINT_BASE_DIR = '../checkpoint/'
ALEXNET_PRETRAINED_WEIGHT_PATH = '../pretrained_weights/alexnet_weights.h5'
VGG16_PRETRAINED_WEIGHT_PATH = '../pretrained_weights/vgg16_weights_th_dim_ordering_th_kernels.h5'
RESNET152_PRETRAINED_WEIGHT_PATH = '../pretrained_weights/resnet152_weights.h5'
DENSENET169_PRETRAINED_WEIGHT_PATH = '../pretrained_weights/densenet169_weights_th.h5'

pretrained_weights_path_map = {ALEXNET_ARCHITECTURE: ALEXNET_PRETRAINED_WEIGHT_PATH,
                               VGG16_ARCHITECTURE: VGG16_PRETRAINED_WEIGHT_PATH,
                               RESNET152_ARCHITECTURE: RESNET152_PRETRAINED_WEIGHT_PATH,
                               DENSENET169_ARCHITECTURE: DENSENET169_PRETRAINED_WEIGHT_PATH}


# Predict label for test data on yearbook dataset
def predictTestYearbookFromModel(model, architecture, sample=False):
    test_list = util.testListYearbook(sample=sample)

    total_count = len(test_list)
    print(get_time_string() + "Total test data: ", total_count)

    test_images = [path.join(YEARBOOK_TEST_PATH, item[0]) for item in test_list]
    processed_test_images = preprocess_image_batch(test_images, architecture)

    output = open(YEARBOOK_TEST_LABEL_PATH, 'w')
    for image in processed_test_images:
        pred_year = np.argmax(model.predict(np.stack([image], axis=0))) + 1900
        out_string = str(pred_year) + '\n'
        output.write(out_string)
    output.close()


if __name__ == "__main__":
    if is_using_gpu():
        print('Program is using GPU..')
    else:
        print('Program is using CPU..')

    parser = ArgumentParser("Evaluate a model on the validation set")
    parser.add_argument("--DATASET_TYPE", dest="dataset_type",
                        help="Dataset: yearbook/geolocation", required=True)
    parser.add_argument("--type", dest="type",
                        help="Dataset: valid/test", required=True)

    parser.add_argument("--model_architecture", dest="model_architecture",
                        help="Model architecture: alexnet/vgg16/resnet152", required=True)
    parser.add_argument("--load_saved_model", dest="load_saved_model",
                        help="load_saved_model: Whether to use saved model",
                        required=False, default=0, type=int)
    parser.add_argument("--initial_epoch", dest="initial_epoch",
                        help="initial_epoch: Epoch to start training from",
                        required=False, default=0, type=int)
    parser.add_argument("--checkpoint_file_name", dest="checkpoint_file_name",
                        help="checkpoint_file_name: h5 file name to save to/load from", required=True)

    parser.add_argument("--use_pretraining", dest="use_pretraining",
                        help="use_pretraining: Whether to use supervised pretraining",
                        required=False, default=1, type=int)
    parser.add_argument("--fine_tuning_method", dest="fine_tuning_method",
                        help="fine_tuning_method: end-to-end/phase-by-phase",
                        required=False)

    parser.add_argument("--batch_size", dest="batch_size",
                        help="batch_size: size of batches while fitting the model",
                        required=False, default=128, type=int)
    parser.add_argument("--num_epochs", dest="num_epochs",
                        help="num_epochs: number of epochs to train the model for",
                        required=False, default=20, type=int)
    parser.add_argument("--loss", dest="loss",
                        help="loss function to use: l1|mse|mae",
                        required=True)
    parser.add_argument("--optimizer", dest="optimizer",
                        help="optimizer to use: sgd|adagrad",
                        required=False, default='sgd')

    parser.add_argument("--sample", dest="sample",
                        help="sample: whether to use sample dataset",
                        required=False, default=0, type=int)

    args = parser.parse_args()
    print('Args provided: ' + str(args))

    print(get_time_string() + 'Operating on ' + args.dataset_type + ' dataset..')

    if args.model_architecture not in ARCHITECTURES:
        raise Exception('Invalid model architecture type!')

    if args.fine_tuning_method is not None and args.fine_tuning_method not in FINE_TUNING_METHODS:
        raise Exception('Invalid fine_tuning_method specified!')

    if not args.checkpoint_file_name.endswith('.h5'):
        raise Exception('Checkpoint file should end with h5 format!')

    if args.dataset_type == 'yearbook':
        model = YearbookModel()
        trained_model = model.getModel(model_architecture=args.model_architecture,
                                       load_saved_model=args.load_saved_model,
                                       model_save_path=CHECKPOINT_BASE_DIR + args.checkpoint_file_name,
                                       use_pretraining=True,
                                       pretrained_weights_path=pretrained_weights_path_map[args.model_architecture],
                                       train_dir=None, val_dir=None, fine_tuning_method=args.fine_tuning_method,
                                       batch_size=args.batch_size, num_epochs=args.num_epochs,
                                       optimizer=args.optimizer, loss=args.loss,
                                       initial_epoch=args.initial_epoch,
                                       sample=args.sample)
        if args.type == 'valid':
            evaluateYearbookFromModel(trained_model, args.model_architecture, args.sample)
        elif args.type == 'test':
            predictTestYearbookFromModel(trained_model, args.model_architecture, args.sample)
        else:
            print(get_time_string() + "Unknown type '%s'", args.type)
    else:
        print(get_time_string() + "only yearbook data supported in customgrade for now!")
        exit(1)
