from Modules.buildingclassifier import BuildingClassifier
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    parser.add_argument(
        "--model_filename", type=str, default='Models/model.h5',
        help = "The location to save the created model.  Default is 'Models/model.h5'.")
    parser.add_argument(
        "--number_categories", type=int, default=3,
        help = "The number of output categories for the model (3 = ternary classifier, 1 = binary classifier).  Default is 3.")
    parser.add_argument(
        "--dense_layer_sizes", type=str, default='[1024, 512, 256]',
        help = "A list of the sizes of the dense layers to be added onto the pretrained model.  Default is '[1024, 512, 256]'.")
    parser.add_argument(
        "--dropout_fraction", type=float, default=0.2,
        help = "The dropout fraction to be used after each dense layer.  Default is 0.2")
    parser.add_argument(
        "--unfrozen_layers", type=str, default=21,
        help = "The number of layers to unfreeze for training.  Default is 21.")
    parser.add_argument(
        "--focal_loss", type=bool, default=False,
        help = "Modify loss function to use focal loss.  Default is False.")
    flags = parser.parse_args()
    model_filename = flags.model_filename
    number_categories = flags.number_categories
    dense_layer_sizes = eval(flags.dense_layer_sizes)
    dropout_fraction = flags.dropout_fraction
    unfrozen_layers = flags.unfrozen_layers
    focal_loss = flags.focal_loss
    image_classifier = BuildingClassifier(model_filename)
    image_classifier.create_inception_model(number_categories, dense_layer_sizes, dropout_fraction, unfrozen_layers, focal_loss)
