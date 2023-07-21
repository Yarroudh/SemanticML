import logging
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import pickle
import laspy
import numpy as np

def read_las_data_in_chunks(filename, features, chunk_size):
    with laspy.open(filename) as file:
        total_points = file.header.point_count
        num_chunks = (total_points + chunk_size - 1) // chunk_size
        fields = [field.name for field in file.header.point_format]
        print(f"Reading {total_points} points in {num_chunks} chunks of size {chunk_size}.")

        X, Y = [], []
        for chunk_points in tqdm(file.chunk_iterator(chunk_size), total=num_chunks, unit="chunk"):
            # Extract the required features from the chunk_points
            chunk_X = np.asarray(np.column_stack([getattr(chunk_points, field) for field in features if field]), dtype=np.float32)

            # Extract Y from classification field if present
            if 'classification' in fields:
                chunk_Y = np.asarray(chunk_points.classification, dtype=np.float32)
            else:
                chunk_Y = None

            # Append the chunk data to X and Y
            X.append(chunk_X)
            if chunk_Y is not None:
                Y.append(chunk_Y)

    X = np.vstack(X)
    Y = np.concatenate(Y) if len(Y) > 0 else None

    return X, Y

def train_model(X_train, Y_train, hyperparameters):
    """
    Trains the model using the specified hyperparameters and input data.

    :param hyperparameters: A dictionary of hyperparameters for the model.
    :type hyperparameters: dict
    :param X_train: The input features for the training data.
    :type X_train: array-like
    :param Y_train: The target labels for the training data.
    :type Y_train: array-like

    :return: The trained classification model.
    :rtype: object
    """
    method = hyperparameters['method']
    del hyperparameters['method']

    if method == "RF":
        model = RandomForestClassifier(**hyperparameters)
    elif method == "GB":
        model = GradientBoostingClassifier(**hyperparameters)

    model.fit(X_train, Y_train)
    return model

def setup_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Log to console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def save_model(model, filename):
    """
    Save the trained machine learning model as a .pkl file.

    :param model: The trained machine learning model.
    :type model: Union[RandomForestClassifier, GradientBoostingClassifier]

    :param filename: The name of the output file.
    :type filename: str

    :return: None
    """
    with open(filename, 'wb') as out:
        pickle.dump(model, out, pickle.HIGHEST_PROTOCOL)

def read_model(filepath):
    """
    Read a machine learning model from a .pkl file.

    :param filepath: The path to the input file.
    :type filepath: str

    :return: The loaded machine learning model.
    :rtype: Union[RandomForestClassifier, GradientBoostingClassifier]
    """
    return pickle.load(open(filepath, 'rb'))