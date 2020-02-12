from buildingclassifier import BuildingClassifier
import os
if __name__ == '__main__':
    model_filename = os.getenv('MODEL_FILENAME')
    epochs = int(os.getenv('EPOCHS'))
    batch_size = int(os.getenv('BATCH_SIZE'))
    training_directory = os.getenv('TRAINING_DIRECTORY')
    test_directory = os.getenv('TEST_DIRECTORY')
    trained_model_filename = os.getenv('TRAINED_MODEL_FILENAME')
    metrics_filename = os.getenv('METRICS_FILENAME')
    binary = int(os.getenv('BINARY'))
    weights = None
    if not(binary):
        weights = eval(os.getenv('WEIGHTS'))

    image_classifier = BuildingClassifier(model_filename)
    image_classifier.train_model(epochs, batch_size, training_directory, test_directory, trained_model_filename, metrics_filename, binary, weights)
