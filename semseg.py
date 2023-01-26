import click
import collections
import os
import json
import laspy
import time
import pickle
import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import jaccard_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.neighbors import KNeighborsClassifier

def train_model(X_train, Y_train, n_estimators, max_depth, n_jobs, criterion, min_samples_split, min_samples_leaf, min_weight_fraction_leaf, max_features, max_leaf_nodes, min_impurity_decrease, bootstrap, oob_score, random_state, verbose, warm_start, class_weight, ccp_alpha, max_samples):
    ''' Train the Random Forest model with the specified parameters and return it
        Attributes:
            X_train (np.array)  :   Training data
            Y_train (np.array)  :   Training classes
            n_estimators (int)  :   Number of trees in the forest
            max_depth (int)     :   Maximum depth of each tree
            n_jobs (int)        :   Number of threads used to train the model
        
        Return:
            model (np.RandomForestClassifier)  :   trained model
    '''
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, criterion=criterion, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, min_weight_fraction_leaf=min_weight_fraction_leaf, max_features=max_features, max_leaf_nodes=max_leaf_nodes, min_impurity_decrease=min_impurity_decrease, bootstrap=bootstrap, oob_score=oob_score, n_jobs=n_jobs, random_state=random_state, verbose=verbose, warm_start=warm_start, class_weight=class_weight, ccp_alpha=ccp_alpha, max_samples=max_samples)
    model.fit(X_train, Y_train)
    return model

def save_model(model, filename):
    ''' Save the trained machine learning model as .pkl file
        Attribures:
            model (np.RandomForestClassifier)   :   Model to save
            filename (string)                   :   Model output file
    '''
    with open(filename, 'wb') as out:
        pickle.dump(model, out, pickle.HIGHEST_PROTOCOL)

def read_model(filepath):
    ''' Read the Random Forest model from a .pkl file
        Attributes:
            filepath (string)   :   Path to the .pkl file
    '''
    return pickle.load(open(filepath, 'rb'))

class OrderedGroup(click.Group):
    def __init__(self, name=None, commands=None, **attrs):
        super(OrderedGroup, self).__init__(name, commands, **attrs)
        self.commands = commands or collections.OrderedDict()

    def list_commands(self, ctx):
        return self.commands

@click.group(cls=OrderedGroup, help="CLI tool to perform semantic segmentation of 3D point clouds using Random Forest algorithm.")
def cli():
    pass

@click.command()
@click.argument('config', type=click.Path(exists=True), required=True)

def train(config):
    '''
    Train the model for semantic segmentation of 3D point clouds.
    '''
    if (os.path.exists("./output")==False):
        os.mkdir("./output")

    if (os.path.exists("./output/model")==False):
        os.mkdir("{./output/model")

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
    
    # RF parameters
    n_estimators = configuration["parameters"]["n_estimators"]
    criterion = configuration["parameters"]["criterion"]
    max_depths = [configuration["parameters"]["max_depths"]]
    min_samples_split = configuration["parameters"]["min_samples_split"]
    min_samples_leaf = configuration["parameters"]["min_samples_leaf"]
    min_weight_fraction_leaf = configuration["parameters"]["min_weight_fraction_leaf"]
    max_features = configuration["parameters"]["max_features"]
    max_leaf_nodes = configuration["parameters"]["max_leaf_nodes"]
    min_impurity_decrease = configuration["parameters"]["min_impurity_decrease"]
    bootstrap = configuration["parameters"]["bootstrap"]
    oob_score = configuration["parameters"]["oob_score"]
    n_jobs = configuration["parameters"]["n_jobs"]
    random_state = configuration["parameters"]["random_state"]
    verbose = configuration["parameters"]["verbose"]
    warm_start = configuration["parameters"]["warm_start"]
    class_weight = configuration["parameters"]["class_weight"]
    ccp_alpha = configuration["parameters"]["ccp_alpha"]
    max_samples = configuration["parameters"]["n_jobs"]

    # Load data
    print('\nLoading data')  
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=42)
    print('\tTraining samples: {}\n\tTesting samples: {}\n\tUsing features: {}'.format(len(Y_train), len(Y_test), features))

    # Train the model
    print('\nTraining the model')
    best_conf = {'ne' : 0, 'md' : 0} # Best configuration initialisation
    best_f1 = 0
    f1_results=[]
    start = time.time()
    for ne, md in list(itertools.product(n_estimators, max_depths)):    # Train the model with different parameters and pick the one having the maximum f1-score on the test-set
        model = train_model(X_train, Y_train, ne, md, criterion=criterion, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, min_weight_fraction_leaf=min_weight_fraction_leaf, max_features=max_features, max_leaf_nodes=max_leaf_nodes, min_impurity_decrease=min_impurity_decrease, bootstrap=bootstrap, oob_score=oob_score, n_jobs=n_jobs, random_state=random_state, verbose=verbose, warm_start=warm_start, class_weight=class_weight, ccp_alpha=ccp_alpha, max_samples=max_samples)   # Train the model
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

    model = train_model(X_train, Y_train, best_conf['ne'], best_conf['md'], criterion=criterion, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, min_weight_fraction_leaf=min_weight_fraction_leaf, max_features=max_features, max_leaf_nodes=max_leaf_nodes, min_impurity_decrease=min_impurity_decrease, bootstrap=bootstrap, oob_score=oob_score, n_jobs=n_jobs, random_state=random_state, verbose=verbose, warm_start=warm_start, class_weight=class_weight, ccp_alpha=ccp_alpha, max_samples=max_samples)
    save_model(model, './output/model/ne{}_md{}.pkl'.format(best_conf['ne'], best_conf['md']))

    end = time.time()
    processTime = end - start

    print('\n\tBest parameters: ne: {}, md: {}'.format(best_conf['ne'], best_conf['md']))
    print('\tFeature importance:\n{}'.format(model.feature_importances_))
    print('\tConfusion matrix:\n{}'.format(confusion_matrix(Y_test, Y_test_pred)))
    print('\tTraining time: {} seconds'.format(time.strftime("%H:%M:%S", time.gmtime(processTime))))


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
        neigh = KNeighborsClassifier(n_neighbors=k)
        neigh.fit(X, Y)
        Y = neigh.predict(X)

    # Save the results in LAS file
    header = file.header
    header.classification = np.asarray(Y)
    las = laspy.LasData(header)
    las.X = file.X
    las.Y = file.Y
    las.Z = file.Z
    las.classification = Y

    # Export results
    las.write("./output/prediction/{}".format(filename))

    end = time.time()
    processTime = end - start
    print('Data classified in: {}'.format(time.strftime("%H:%M:%S", time.gmtime(processTime))))


cli.add_command(train)
cli.add_command(predict)

if __name__ == '__main__':
    cli(prog_name='semseg')