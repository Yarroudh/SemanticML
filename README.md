# Semantic segmentation of 3D point clouds using Random Forest algorithm
*CLI tool to train a model using Random Forest algorithm then use it to perform semantic segmentation of 3D point clouds.*

![image](https://user-images.githubusercontent.com/72500344/214625563-d048d13f-b1d5-42c4-afd0-db906ca9f93e.png)

Semantic segmentation of point clouds is a process of classifying each point in a point cloud into different semantic categories, such as building, road, or vegetation. One approach to accomplish this is by using the Random Forest algorithm. Random Forest is a type of ensemble learning method that combines multiple decision trees to make predictions. In the context of semantic segmentation of point clouds, each decision tree in the Random Forest model would be trained to classify a point based on its features, such as its location and color. The final prediction for a point would be the majority vote of all the decision trees in the forest. Random Forest has been shown to be effective for semantic segmentation of point clouds due to its ability to handle high-dimensional and noisy data.

## Installation

The easiest way to install <code>Semseg</code> on Windows is to use the binary package on the Release page. In case you can not use the Windows installer, or if you are using a different operating system, you can build everything from source.

## Usage of the CLI

After installation, you have a small program called <code>semseg</code>. Use <code>semseg --help</code> to see the detailed help:

```
Usage: semseg [OPTIONS] COMMAND [ARGS]...

  CLI tool to perform semantic segmentation of 3D point clouds using Random
  Forest algorithm.

Options:
  --help  Show this message and exit.

Commands:
  train    Train the model for semantic segmentation of 3D point clouds.   
  predict  Perform semantic segmentation using pre-trained model.
```

The process consists of two distinct steps or <code>commands</code> :

### Step 1 : Model training using labeled data

Model training is the process of using a set of labeled data, known as the training dataset, to adjust the parameters of Random Forest algorithm so that it can make accurate predictions on new, unseen data. The process of training a model involves providing the model with input-output pairs, where the input represents the features of the data and the output represents the desired label or prediction. The model then adjusts its internal parameters, or weights, to minimize the difference between its predictions and the true labels. This process is repeated multiple times, using a technique called backpropagation and optimization algorithms such as gradient descent. The goal is to find the set of parameters that result in the lowest prediction error on the training data.

This is done using the first command <code>train</code>. Use <code>semseg train --help</code> to see the detailed help:

```
Usage: semseg train [OPTIONS] CONFIG

  Train the model for semantic segmentation of 3D point clouds.

Options:
  --help  Show this message and exit.
```

The input data is a LAS file with specified features and <code>classification</code> field that respresents the label. The command takes one argument which is a <code>JSON</code> file that contains the features to use, the training data path and the algorithm parameters, as shown in this example:

```json
{
    "features": ["green", "Verticality16", "Verticality8", "Linearity16", "Linearity8", "Planarity16", "Planarity8", "Surfacevariation5", "Numberneighbors10"],
    "label": ["Ground", "Vegetation", "Rail", "Catenary pole", "Cable", "Infrastructure"],
    "training_data": "C:/Users/Administrateur/Desktop/railway.las",
    "parameters": {
        "n_estimators": [20, 40, 60],
        "criterion": "gini",
        "max_depths": null,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "min_weight_fraction_leaf": 0,
        "max_features": "sqrt",
        "max_leaf_nodes": null,
        "min_impurity_decrease": 0.0,
        "bootstrap": true,
        "oob_score": false,
        "n_jobs": null,
        "random_state": null,
        "verbose": 0,
        "warm_start": false,
        "class_weight": null,
        "ccp_alpha": 0.0,
        "max_samples": null
    }
}
```

### Step 2 : Model training using labeled data

Once the model is trained, it can be used to make predictions on new, unseen data.
