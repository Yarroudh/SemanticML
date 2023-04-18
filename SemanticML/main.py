import click
import collections
import os
import json
import laspy
import time
import pickle
import itertools
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import jaccard_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

def train_model(method, X_train, Y_train, **kwargs):
    '''
    Trains the model using the specified method and input data.

    :param method: The name of the classification method to use. Currently supported
                   options are "RandomForest" and "GradientBoosting".
    :type method: str
    :param X_train: The input features for the training data.
    :type X_train: array-like
    :param Y_train: The target labels for the training data.
    :type Y_train: array-like
    :param kwargs: Additional keyword arguments that are passed on to the classification
                   model constructor.
    :type kwargs: dict

    :return: The trained classification model.
    :rtype: object
    '''
    if (method == "RandomForest"):
        n_estimators = kwargs.get('n_estimators', None)
        max_depth = kwargs.get('max_depth', None)
        n_jobs = kwargs.get('n_jobs', None)
        criterion = kwargs.get('criterion', None)
        min_samples_split = kwargs.get('min_samples_split', None)
        min_samples_leaf = kwargs.get('min_samples_leaf', None)
        min_weight_fraction_leaf = kwargs.get('min_weight_fraction_leaf', None)
        max_features = kwargs.get('max_features', None)
        max_leaf_nodes = kwargs.get('max_leaf_nodes', None)
        min_impurity_decrease = kwargs.get('min_impurity_decrease', None)
        bootstrap = kwargs.get('bootstrap', None)
        oob_score = kwargs.get('oob_score', None)
        random_state = kwargs.get('random_state', None)
        verbose = kwargs.get('verbose', None)
        warm_start = kwargs.get('warm_start', None)
        class_weight = kwargs.get('class_weight', None)
        ccp_alpha = kwargs.get('ccp_alpha', None)
        max_samples = kwargs.get('max_samples', None)

        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, criterion=criterion, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, min_weight_fraction_leaf=min_weight_fraction_leaf, max_features=max_features, max_leaf_nodes=max_leaf_nodes, min_impurity_decrease=min_impurity_decrease, bootstrap=bootstrap, oob_score=oob_score, n_jobs=n_jobs, random_state=random_state, verbose=verbose, warm_start=warm_start, class_weight=class_weight, ccp_alpha=ccp_alpha, max_samples=max_samples)
    
    elif (method == "GradientBoosting"):
        loss = kwargs.get('loss', None)
        learning_rate = kwargs.get('learning_rate', None)
        n_estimators = kwargs.get('n_estimators', None)
        subsample = kwargs.get('subsample', None)
        criterion = kwargs.get('criterion', None)
        min_samples_split = kwargs.get('min_samples_split', None)
        min_samples_leaf = kwargs.get('min_samples_leaf', None)
        min_weight_fraction_leaf = kwargs.get('min_weight_fraction_leaf', None)
        max_depth = kwargs.get('max_depths', None)
        min_impurity_decrease = kwargs.get('min_impurity_decrease', None)
        init = kwargs.get('init', None)
        random_state = kwargs.get('random_state', None)
        max_features = kwargs.get('max_features', None)
        verbose = kwargs.get('verbose', None)
        max_leaf_nodes = kwargs.get('max_leaf_nodes', None)
        warm_start = kwargs.get('warm_start', None)
        validation_fraction = kwargs.get('validation_fraction', None)
        n_iter_no_change = kwargs.get('n_iter_no_change', None)
        tol = kwargs.get('tol', None)
        ccp_alpha = kwargs.get('ccp_alpha', None)

        model = GradientBoostingClassifier(loss=loss, learning_rate=learning_rate, n_estimators=n_estimators, subsample=subsample, criterion=criterion, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, min_weight_fraction_leaf=min_weight_fraction_leaf, max_depth=max_depth, min_impurity_decrease=min_impurity_decrease, init=init, random_state=random_state, max_features=max_features, verbose=verbose, max_leaf_nodes=max_leaf_nodes, warm_start=warm_start, validation_fraction=validation_fraction, n_iter_no_change=n_iter_no_change, tol=tol, ccp_alpha=ccp_alpha)

    model.fit(X_train, Y_train)
    return model

def save_model(model, filename):
    '''
    Save the trained machine learning model as a .pkl file.

    :param model: The trained machine learning model.
    :type model: Union[RandomForestClassifier, GradientBoostingClassifier]

    :param filename: The name of the output file.
    :type filename: str

    :return: None
    '''
    with open(filename, 'wb') as out:
        pickle.dump(model, out, pickle.HIGHEST_PROTOCOL)

def read_model(filepath):
    '''
    Read a machine learning model from a .pkl file.

    :param filepath: The path to the input file.
    :type filepath: str

    :return: The loaded machine learning model.
    :rtype: Union[RandomForestClassifier, GradientBoostingClassifier]
    '''
    return pickle.load(open(filepath, 'rb'))

class OrderedGroup(click.Group):
    def __init__(self, name=None, commands=None, **attrs):
        super(OrderedGroup, self).__init__(name, commands, **attrs)
        self.commands = commands or collections.OrderedDict()

    def list_commands(self, ctx):
        return self.commands

@click.group(cls=OrderedGroup, help="CLI tool to perform semantic segmentation of 3D point clouds using Machine Learning algorithms.")
def cli():
    pass

@click.command()
@click.argument('config', type=click.Path(exists=True), required=True)
@click.option('--method', help='Learning method for classification.', type=click.Choice(['RandomForest', 'GradientBoosting']), default="RandomForest", required=False, show_default=True)

def train(config, method):
    '''
    Train the model for semantic segmentation of 3D point clouds.
    '''
    if (os.path.exists("./output")==False):
        os.mkdir("./output")

    if (os.path.exists("./output/model")==False):
        os.mkdir("./output/model")

    with open(config) as file:
        configuration = json.load(file)


    # Read train and validation data from a file
    debug = True

    file = laspy.read(configuration["training_data"])
    features = configuration["features"]

    fields = [field.name for field in file.point_format]
    if ('classification' in fields):
        Y = np.asarray(file.classification, dtype=np.float32)
        fields.remove('classification')

    X = np.asarray(np.column_stack([getattr(file, field) for field in features]), dtype=np.float32)

    # Load data
    print('\nLoading data')  
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    print('\tTraining samples: {}\n\tTesting samples: {}\n\tUsing features: {}'.format(len(Y_train), len(Y_test), features))

    if (method == "RandomForest"):
        # RF parameters
        n_estimators = configuration["parameters"]["RandomForest"]["n_estimators"]
        criterion = configuration["parameters"]["RandomForest"]["criterion"]
        max_depths = [configuration["parameters"]["RandomForest"]["max_depths"]]
        min_samples_split = configuration["parameters"]["RandomForest"]["min_samples_split"]
        min_samples_leaf = configuration["parameters"]["RandomForest"]["min_samples_leaf"]
        min_weight_fraction_leaf = configuration["parameters"]["RandomForest"]["min_weight_fraction_leaf"]
        max_features = configuration["parameters"]["RandomForest"]["max_features"]
        max_leaf_nodes = configuration["parameters"]["RandomForest"]["max_leaf_nodes"]
        min_impurity_decrease = configuration["parameters"]["RandomForest"]["min_impurity_decrease"]
        bootstrap = configuration["parameters"]["RandomForest"]["bootstrap"]
        oob_score = configuration["parameters"]["RandomForest"]["oob_score"]
        n_jobs = configuration["parameters"]["RandomForest"]["n_jobs"]
        random_state = configuration["parameters"]["RandomForest"]["random_state"]
        verbose = configuration["parameters"]["RandomForest"]["verbose"]
        warm_start = configuration["parameters"]["RandomForest"]["warm_start"]
        class_weight = configuration["parameters"]["RandomForest"]["class_weight"]
        ccp_alpha = configuration["parameters"]["RandomForest"]["ccp_alpha"]
        max_samples = configuration["parameters"]["RandomForest"]["max_samples"]

    elif (method == "GradientBoosting"):
        n_estimators = configuration["parameters"]["GradientBoosting"]["n_estimators"]
        loss = configuration["parameters"]["GradientBoosting"]["loss"]
        learning_rate = configuration["parameters"]["GradientBoosting"]["learning_rate"]
        subsample = configuration["parameters"]["GradientBoosting"]["subsample"]
        criterion = configuration["parameters"]["GradientBoosting"]["criterion"]
        min_samples_split = configuration["parameters"]["GradientBoosting"]["min_samples_split"]
        min_samples_leaf = configuration["parameters"]["GradientBoosting"]["min_samples_leaf"]
        min_weight_fraction_leaf = configuration["parameters"]["GradientBoosting"]["min_weight_fraction_leaf"]
        max_depths = configuration["parameters"]["GradientBoosting"]["max_depths"]
        min_impurity_decrease = configuration["parameters"]["GradientBoosting"]["min_impurity_decrease"]
        init = configuration["parameters"]["GradientBoosting"]["init"]
        random_state = configuration["parameters"]["GradientBoosting"]["random_state"]
        max_features = configuration["parameters"]["GradientBoosting"]["max_features"]
        verbose = configuration["parameters"]["GradientBoosting"]["verbose"]
        max_leaf_nodes = configuration["parameters"]["GradientBoosting"]["max_leaf_nodes"]
        warm_start = configuration["parameters"]["GradientBoosting"]["warm_start"]
        validation_fraction = configuration["parameters"]["GradientBoosting"]["validation_fraction"]
        n_iter_no_change = configuration["parameters"]["GradientBoosting"]["n_iter_no_change"]
        tol = configuration["parameters"]["GradientBoosting"]["tol"]
        ccp_alpha = configuration["parameters"]["GradientBoosting"]["ccp_alpha"]

    # Scale the features using StandardScaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train the model
    print('\nTraining the model')
    best_conf = {'ne' : 0, 'md' : 0} # Best configuration initialisation
    best_f1 = 0
    f1_results=[]
    start = time.time()
    for ne, md in list(itertools.product(n_estimators, max_depths)): # Train the model with different parameters and pick the one having the maximum f1-score on the test-set
        # Train the model
        if (method == "RandomForest"):
            model = train_model(method, X_train, Y_train, n_estimators=ne, max_depth=md, criterion=criterion, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, min_weight_fraction_leaf=min_weight_fraction_leaf, max_features=max_features, max_leaf_nodes=max_leaf_nodes, min_impurity_decrease=min_impurity_decrease, bootstrap=bootstrap, oob_score=oob_score, n_jobs=n_jobs, random_state=random_state, verbose=verbose, warm_start=warm_start, class_weight=class_weight, ccp_alpha=ccp_alpha, max_samples=max_samples)
        elif (method == "GradientBoosting"):
            model = train_model(method, X_train, Y_train, n_estimators=ne, max_depth=md, loss=loss, learning_rate=learning_rate, subsample=subsample, criterion=criterion, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, min_weight_fraction_leaf=min_weight_fraction_leaf, min_impurity_decrease=min_impurity_decrease, init=init, random_state=random_state, max_features=max_features, verbose=verbose, max_leaf_nodes=max_leaf_nodes, warm_start=warm_start, validation_fraction=validation_fraction, n_iter_no_change=n_iter_no_change, tol=tol, ccp_alpha=ccp_alpha)
        
        Y_test_pred = model.predict(X_test)  # Test the model, using only the specified features
            
        acc = accuracy_score(Y_test, Y_test_pred)    # Compute metrics and update best model
        f1 = f1_score(Y_test, Y_test_pred, average='weighted')
        f1_results.append(f1)
        recall=recall_score(Y_test, Y_test_pred, average='weighted')
        precision= precision_score(Y_test, Y_test_pred, average='weighted')
        js=jaccard_score(Y_test, Y_test_pred, average='weighted')
        
        if f1 > best_f1: # Update best configuration
            best_conf['ne'] = ne
            best_conf['md'] = md
            best_f1 = f1
            
        if debug: print('\tne: {}, md: {} - acc: {} f1: {} precision:{} recall:{} js: {} oob_score: {}'.format(ne, md, acc, f1, precision, recall, js, model.oob_score))

    if (len(n_estimators) == 1):
        pass
    else:
        model = train_model(X_train, Y_train, best_conf['ne'], best_conf['md'], criterion=criterion, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, min_weight_fraction_leaf=min_weight_fraction_leaf, max_features=max_features, max_leaf_nodes=max_leaf_nodes, min_impurity_decrease=min_impurity_decrease, bootstrap=bootstrap, oob_score=oob_score, n_jobs=n_jobs, random_state=random_state, verbose=verbose, warm_start=warm_start, class_weight=class_weight, ccp_alpha=ccp_alpha, max_samples=max_samples)
    
    save_model(model, './output/model/ne{}_md{}.pkl'.format(best_conf['ne'], best_conf['md']))

    end = time.time()
    processTime = end - start

    print('\n\tBest parameters: ne: {}, md: {}'.format(n_estimators, max_depths))
    print('\tFeature importance:\n{}'.format(model.feature_importances_))
    print('\tConfusion matrix:\n{}'.format(confusion_matrix(Y_test, Y_test_pred)))
    print('\tTraining time: {} seconds'.format(time.strftime("%H:%M:%S", time.gmtime(processTime))))


@click.command()
@click.argument('config', type=click.Path(exists=True), required=True)
@click.argument('pointcloud', type=click.Path(exists=True), required=True)
@click.argument('model', type=click.Path(exists=True), required=True)
@click.option('--filename', help='Write the evaluation results to .CSV file.', type=click.Path(exists=False), default='output/evaluation.csv', show_default=True)

def evaluate(config, pointcloud, model, filename):
    pass

@click.command()
@click.argument('config', type=click.Path(exists=True), required=True)
@click.argument('pointcloud', type=click.Path(exists=True), required=True)
@click.argument('model', type=click.Path(exists=True), required=True)
@click.option('--regularize', help='If checked the input data will be regularized.', type=bool, default=False, required=False, show_default=True)
@click.option('-k', help='Number of neighbors to use if regularization is set.', type=click.INT, default=10, required=False, show_default=True)
@click.option('--filename', help='Write the classified point cloud in a .LAS file.', type=click.Path(exists=False), required=True, show_default=True)

def predict(config, pointcloud, model, regularize, k, filename):
    '''
    Perform semantic segmentation using pre-trained model.
    '''
    if (os.path.exists("./output")==False):
        os.mkdir("./output")

    if (os.path.exists("./output/prediction")==False):
        os.mkdir("./output/prediction")

    with open(config) as file:
        configuration = json.load(file)

    features = configuration["features"]

    start = time.time()
    # Load the model
    model = read_model(model)

    # Read data
    file = laspy.read(pointcloud)
    X = np.asarray(np.column_stack([getattr(file, field) for field in features]), dtype=np.float32)

    # Perform semantic segmentation
    print ('Classifying the dataset')
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
    las.classification = Y

    # Export results
    las.write("./output/prediction/{}".format(filename))

    end = time.time()
    processTime = end - start
    print('Data classified in: {}'.format(time.strftime("%H:%M:%S", time.gmtime(processTime))))


cli.add_command(train)
cli.add_command(predict)

if __name__ == '__main__':
    cli(prog_name='sml')