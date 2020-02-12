from buildingclassifier import BuildingClassifier

if __name__ == '__main__':
    model_filename = 'Models/test_model.h5'
    number_categories = 3
    dense_layer_sizes = [1024, 512, 256]
    dropout_fraction = 0.2
    unfrozen_layers = 21
    focal_loss = False
    image_classifier = BuildingClassifier(model_filename)
    image_classifier.create_inception_model(number_categories, dense_layer_sizes, dropout_fraction, unfrozen_layers, focal_loss)
