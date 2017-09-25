class Model:
    ARCHITECTURES = ['alextnet', 'vgg16']
    get_model_function = {}

    def __init__(self):
        self.get_model_function['alexnet'] = self.getAlexNet
        self.get_model_function['vgg16'] = self.getVGG16

    def getModel(self, model_architecture, load_pretrained, weights_path, train_dir, val_dir, use_pre_training,
                 fine_tuning_method):
        if model_architecture not in self.ARCHITECTURES:
            raise 'Invalid architecture name!'
        return self.get_model_function[model_architecture](load_pretrained, weights_path, train_dir, val_dir,
                                                           use_pre_training, fine_tuning_method)

    def getAlexNet(self, load_pretrained, weights_path, train_dir, val_dir, use_pre_training, fine_tuning_method):
        return None

    def getVGG16(self, load_pretrained, weights_path, train_dir, val_dir, use_pre_training, fine_tuning_method):
        return None
