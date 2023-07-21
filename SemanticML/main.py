import click
import collections
import os
import json
import laspy
import time
import itertools
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import jaccard_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
from utils import read_las_data_in_chunks, train_model, setup_logger, save_model, read_model


class OrderedGroup(click.Group):
    def __init__(self, name=None, commands=None, **attrs):
        super(OrderedGroup, self).__init__(name, commands, **attrs)
        self.commands = commands or collections.OrderedDict()

    def list_commands(self, ctx):
        return self.commands

@click.group(cls=OrderedGroup, help="A package for semantic segmentation of 3D point clouds using Machine Learning algorithms.")
def cli():
    pass

@click.command()
@click.argument('config', type=click.Path(exists=True), required=True)
@click.option('--method', '-m', help='Learning method for classification.', type=click.Choice(['RF', 'GB']), default="RF", required=False, show_default=True)
@click.option('--output', '-o', help='Output folder.', type=click.Path(exists=False), required=False, default='output', show_default=True)

def train(config, method, output):
    """
    Train the model for semantic segmentation of 3D point clouds.
    """
    if not os.path.exists(output):
        os.mkdir(output)

    if not os.path.exists(f"{output}/model"):
        os.mkdir(f"{output}/model")

    with open(config) as file:
        configuration = json.load(file)

    # Initialize logger
    logger = setup_logger()

    # Read data
    logger.info("Reading data..")
    debug = True
    training_data = configuration["training_data"]
    features = configuration["features"]
    chunk_size = configuration["chunk_size"]

    if chunk_size == -1:
        file = laspy.read(training_data)
        fields = [field.name for field in file.point_format]
        if ('classification' in fields):
            Y = np.asarray(file.classification, dtype=np.float32)
            fields.remove('classification')
        X = np.asarray(np.column_stack([getattr(file, field) for field in features]), dtype=np.float32)

    else:
        X, Y = read_las_data_in_chunks(training_data, features, chunk_size)

    logger.info("Data read.")

    # Split train/test data
    test_size = configuration["test_size"]
    random_state = configuration["random_state"]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)
    print('\tTraining samples: {}\n\tTesting samples: {}\n\tUsing features: {}'.format(len(Y_train), len(Y_test), features))

    # Hyperparameters dictionary
    algorithm = "RandomForest" if method == "RF" else "GradientBoosting"
    hyperparameters = configuration["parameters"].get(algorithm, {})
    hyperparameters['method'] = method

    # Scale the features using StandardScaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train the model with hyperparameter tuning
    n_estimators = hyperparameters['n_estimators']
    max_depths = [hyperparameters['max_depth']] if hyperparameters['max_depth'] is None else hyperparameters['max_depth']

    best_conf = {'ne': 0, 'md': 0} # Best configuration initialisation
    best_f1 = 0

    f1_results=[]
    start = time.time()
    print('\n')
    logger.info("Training process started.")

    for ne, md in list(itertools.product(n_estimators, max_depths)): # Train the model with different parameters and pick the one having the maximum f1-score on the test-set
        # Train the model
        hyperparameters['n_estimators'] = ne
        hyperparameters['max_depth'] = md

        if method == "RF":
            model = train_model(X_train, Y_train, hyperparameters)
        elif method == "GB":
            model = train_model(X_train, Y_train, hyperparameters)

        Y_test_pred = model.predict(X_test)  # Test the model, using only the specified features

        f1 = f1_score(Y_test, Y_test_pred, average='weighted')
        f1_results.append(f1)

        if f1 > best_f1: # Update best configuration
            best_conf['ne'] = ne
            best_conf['md'] = md
            best_f1 = f1

        if debug:
            acc = accuracy_score(Y_test, Y_test_pred)
            recall=recall_score(Y_test, Y_test_pred, average='weighted')
            precision= precision_score(Y_test, Y_test_pred, average='weighted')
            js=jaccard_score(Y_test, Y_test_pred, average='weighted')
            print('\ne: {}, md: {} - acc: {} f1: {} precision:{} recall:{} js: {} oob_score: {}'.format(ne, md, acc, f1, precision, recall, js, model.oob_score))

    if len(n_estimators) > 1:
        model = train_model(X_train, Y_train, best_conf['ne'], best_conf['md'], hyperparameters)

    # Save the model
    save_model(model, f"{output}/model/ne{best_conf['ne']}_md{best_conf['md']}.pkl")
    logger.info("Training process finished.")

    end = time.time()
    processTime = end - start

    print('\n\tBest parameters: ne: {}, md: {}'.format(n_estimators, max_depths))
    print('\tFeature importance:\n{}'.format(model.feature_importances_))
    print('\tConfusion matrix:\n{}'.format(confusion_matrix(Y_test, Y_test_pred)))
    print('\tTraining time: {} seconds'.format(time.strftime("%H:%M:%S", time.gmtime(processTime))))


@click.command()
@click.argument('config', type=click.Path(exists=True), required=True)
@click.argument('data', type=click.Path(exists=True), required=True)
@click.argument('model', type=click.Path(exists=True), required=True)
@click.option('--outfile', help='Write the output as .las/.laz file.', type=click.Path(exists=False), required=False, default='predict.las', show_default=True)
@click.option('--metrics', help='Write the metrics as .txt file.', type=click.Path(exists=False), required=False, default='metrics.txt', show_default=True)
@click.option('--cm', help='Write the confusion matrix as .txt file.', type=click.Path(exists=False), required=False, default='confusion_matrix.txt', show_default=True)
@click.option('--cmfig', help='Write the confusion matrix as .png file.', type=click.Path(exists=False), required=False, default='confusion_matrix.png', show_default=True)
@click.option('--output', '-o', help='Output folder.', type=click.Path(exists=False), required=False, default='output', show_default=True)

def evaluate(config, data, model, outfile, metrics, cm, cmfig, output):
    """
    Evaluate the trained model.
    """
    if not os.path.exists(output):
        os.mkdir(output)

    if not os.path.exists(f"{output}/evaluation"):
        os.mkdir(f"{output}/evaluation")

    with open(config) as file:
        configuration = json.load(file)

    features = configuration["features"]

    start = time.time()

    # Load the model
    model = read_model(model)

    # Read data
    file = laspy.read(data)
    X = np.asarray(np.column_stack([getattr(file, field) for field in features]), dtype=np.float32)
    y_true = np.asarray(file.classification, dtype=np.float32)

    # Perform semantic segmentation
    print ('Classifying data..')
    y_predict = model.predict(X)

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_predict)
    precision = precision_score(y_true, y_predict, average='weighted')
    recall = recall_score(y_true, y_predict, average='weighted')
    f1 = f1_score(y_true, y_predict, average='weighted')
    jaccard = jaccard_score(y_true, y_predict, average='weighted')

    # Calculate confusion matrix
    conf_matrix = confusion_matrix(y_true, y_predict)

    metrics = [['Accuracy', accuracy],
            ['Precision', precision],
            ['Recall', recall],
            ['F1 Score', f1],
            ['IoU', jaccard]]

    print("\n")
    print(tabulate(metrics, headers=["Metrics", "Values"], tablefmt='fancy_grid'))

    # Save the metrics to a txt file
    np.savetxt(metrics, np.column_stack((['Accuracy', 'Precision', 'Recall', 'F1', 'IoU'], [accuracy, precision, recall, f1, jaccard])), delimiter=' ', fmt='%s')
    print(f"\nMetrics saved as {output}/{metrics}")

    # Save the confusion matrix to a txt file
    np.savetxt(cm, conf_matrix, delimiter=',', fmt='%d')
    print(f"\nConfusion matrix saved as {output}/{cm}")

    # Save the classification results in LAS file
    header = file.header
    las = laspy.LasData(header)
    las.points = file.points
    las.prediction = y_predict.astype(np.uint8)
    las.write(f"{output}/evaluation/{outfile}")

    # Plot the confusion matrix
    class_labels = configuration["label"]

    # Create a heatmap of the confusion matrix
    fig, ax = plt.subplots()
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)

    # Set axis labels
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')

    # Save the figure as an image
    plt.savefig(cmfig)
    plt.show()
    print(f"\nFigure saved as {output}/{cmfig}")


@click.command()
@click.argument('config', type=click.Path(exists=True), required=True)
@click.argument('data', type=click.Path(exists=True), required=True)
@click.argument('model', type=click.Path(exists=True), required=True)
@click.option('--regularize', '-r', help='If checked the input data will be regularized.', type=bool, default=False, required=False, show_default=True)
@click.option('-k', help='Number of neighbors to use if regularization is set.', type=click.INT, default=10, required=False, show_default=True)
@click.option('--outfile', help='Write the classified point cloud as .las/.laz file.', type=click.Path(exists=False), required=False, default='classified.las', show_default=True)
@click.option('--output', '-o', help='Output folder.', type=click.Path(exists=False), required=False, default='output', show_default=True)

def predict(config, data, model, regularize, k, outfile, output):
    """
    Perform semantic segmentation using pre-trained model.
    """
    if not os.path.exists(output):
        os.mkdir(output)

    if not os.path.exists(f"{output}/prediction"):
        os.mkdir(f"{output}/prediction")

    with open(config) as file:
        configuration = json.load(file)

    features = configuration["features"]

    start = time.time()
    # Load the model
    model = read_model(model)

    # Read data
    file = laspy.read(data)
    X = np.asarray(np.column_stack([getattr(file, field) for field in features]), dtype=np.float32)

    # Perform semantic segmentation
    print ('Classifying data..')
    Y = model.predict(X)

    # Regularization
    if (regularize):
        neigh = KNeighborsClassifier(n_neighbors=k, algorithm='kd_tree', n_jobs=-1)
        neigh.fit(X, Y)
        Y = neigh.predict(X)

    # Save the results in LAS file
    header = file.header
    las = laspy.LasData(header)
    las.points = file.points
    las.classification = Y.astype(np.uint8)

    # Export results
    las.write(f"{output}/prediction/{outfile}")

    end = time.time()
    processTime = end - start
    print('Data classified in: {}'.format(time.strftime("%H:%M:%S", time.gmtime(processTime))))


cli.add_command(train)
cli.add_command(evaluate)
cli.add_command(predict)

if __name__ == '__main__':
    cli(prog_name='sml')
