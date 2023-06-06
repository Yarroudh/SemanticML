# Semantic segmentation of 3D point clouds using Machine Learning algorithms
*A package for semantic segmentation of 3D unorganized point clouds using Machine Learning techniques.*

![image](https://user-images.githubusercontent.com/72500344/215734559-efe11019-e88d-4d94-9643-ee00e0be13c9.png)

Semantic segmentation of point clouds is the process of classifying each point in a point cloud into different semantic classes, such as building, road, or vegetation. One approach to accomplish this is by using Machine Learning techniques such as Random Forest and Gradient Boosting classifiers. 

**Random Forest** is a type of ensemble learning method that combines multiple decision trees to make predictions. In the context of semantic segmentation of point clouds, each decision tree in the Random Forest model would be trained to classify a point based on its features, such as its location and color. The final prediction for a point would be the majority vote of all the decision trees in the forest. Random Forest has been shown to be effective for semantic segmentation of point clouds due to its ability to handle high-dimensional and noisy data.

![image](https://user-images.githubusercontent.com/72500344/214625563-d048d13f-b1d5-42c4-afd0-db906ca9f93e.png)

**Gradient Boosting** is another Machine Learning technique used for classification tasks. It gives a prediction model in the form of an ensemble of weak prediction models, which are typically decision trees. When a decision tree is the weak learner, the resulting algorithm is called gradient-boosted trees; it usually outperforms random forest. A gradient-boosted trees model is built in a stage-wise fashion as in other boosting methods, but it generalizes the other methods by allowing optimization of an arbitrary differentiable loss function.

![image](https://user-images.githubusercontent.com/72500344/215735009-27690651-0e7b-4597-b426-a9e67d7d7baa.png)

## Installation

You can install SemanticML from source by easily running:

```bash
git clone https://github.com/Yarroudh/SemanticML
cd SemanticML
python setup.py install
```

## Usage of the CLI

After installation, you have a small program called <code>sml</code>. Use <code>sml --help</code> to see the detailed help:

```
Usage: sml [OPTIONS] COMMAND [ARGS]...

  CLI tool to perform semantic segmentation of 3D point clouds using Machine
  Learning techniques.

Options:
  --help  Show this message and exit.

Commands:
  train    Train the model for semantic segmentation of 3D point clouds.    
  predict  Perform semantic segmentation using pre-trained model.
```

The process consists of two distinct steps or <code>commands</code> :

### Step 1 : Model training using labeled data

Model training is the process of using a set of labeled data, known as the training dataset, to adjust the parameters of Random Forest algorithm so that it can make accurate predictions on new, unseen data. The process of training a model involves providing the model with input-output pairs, where the input represents the features of the data and the output represents the desired label or prediction. The model then adjusts its internal parameters to minimize the difference between its predictions and the true labels. The goal is to find the set of parameters that result in the lowest prediction error on the training data.

This is done using the first command <code>train</code>. Use <code>sml train --help</code> to see the detailed help:

```
Usage: sml train [OPTIONS] CONFIG

  Train the model for semantic segmentation of 3D point clouds.      

Options:
  --method [RandomForest|GradientBoosting]
                                  Learning method for classification.
                                  [default: RandomForest]
  --help                          Show this message and exit.
```

The input data is a LAS file with specified features and <code>classification</code> field that respresents the label. The command takes one argument which is a <code>JSON</code> file that contains the features to use, the training data path and the algorithm parameters, as shown in this example:

```json
{
    "features": ["red", "blue", "green", "Verticality16", "Verticality8", "Linearity16", "Linearity8", "Planarity16", "Planarity8", "Surfacevariation5", "Numberneighbors10"],
    "label": ["Ground", "Vegetation", "Rail", "Catenary pole", "Cable", "Infrastructure"],
    "training_data": "C:/Users/Administrateur/Desktop/railway.las",
    "parameters": {
        "RandomForest": {
            "n_estimators": [50],
            "criterion": "entropy",
            "max_depths": null,
            "min_samples_split": 4,
            "min_samples_leaf": 4,
            "min_weight_fraction_leaf": 0,
            "max_features": "sqrt",
            "max_leaf_nodes": null,
            "min_impurity_decrease": 0.0,
            "bootstrap": true,
            "oob_score": false,
            "n_jobs": -1,
            "random_state": null,
            "verbose": 0,
            "warm_start": false,
            "class_weight": null,
            "ccp_alpha": 0.0,
            "max_samples": null
        },
        "GradientBoosting": {
            "n_estimators": [50],
            "loss":"log_loss",
            "learning_rate":0.1,
            "subsample": 1.0,
            "criterion": "friedman_mse",
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "min_weight_fraction_leaf": 0.0,
            "max_depths": [3],
            "min_impurity_decrease": 0.0,
            "init": null,
            "random_state": null,
            "max_features": null,
            "verbose": 0,
            "max_leaf_nodes": null,
            "warm_start": false,
            "validation_fraction": 0.1,
            "n_iter_no_change": null,
            "tol": 1e-4,
            "ccp_alpha": 0.0
        }
    }
}
```

Thus, the command for Random Forest algorithm could be:

```
sml train --method RandomForest config.json
```

Or simply:

```
sml train config.json
```

You can also choose to use Gradient Boosting classifier by typing:

```
sml train --method GradientBoosting config.json
```

The output is the model with the best parameters, saved as a <code>pickle</code> file in <code>./output/model</code>.
Pickling a model and saving it to disk allows you to save the state of the model, including all its trained parameters, so that it can be loaded and used again later without having to retrain the model from scratch. This is useful in cases where training a model takes a long time, or if you want to share a trained model with others.

### Step 2 : Predictions

Once the model is trained, it can be used to make predictions on new, unseen data. This is done by providing the model with input data, and the model generates an output, which is a <code>LAS</code> file with <code>classification</code> as a new scalar field.

This is done using the second command <code>predict</code>. Use <code>sml predict --help</code> to see the detailed help:

```
Usage: sml predict [OPTIONS] CONFIG POINTCLOUD MODEL

  Perform semantic segmentation using pre-trained model.

Options:
  --regularize BOOLEAN  If checked the input data will be regularized.      
                        [default: False]
  -k INTEGER            Number of neighbors to use if regularization is set.
                        [default: 10]
  --filename PATH       Write the classified point cloud in a .LAS file.    
                        [required]
  --help                Show this message and exit.
```

#### Basic usage

```
sml predict config.json unclassified.las ./output/model/ne60_mdNone.pkl --filename classified.las
```

This uses the trained model stored as pickle file <code>ne60_mdNone.pkl</code> to perfoem semantic segmentation on <code>unclassified.las</code>. The output is named <code>classified.las</code> and can be found in the folder <code>./output/prediction</code>.

#### Regularization

The rendering of the Random Forest can be improved by an algorithm that reduces the noise. The principle of the algorithm is quite simple:
- For each point, we calculate the K-nearest neighbors 
- We calculate the percentage of each class among these K-nearest neighbors 
- If the majority class is present in the environment at more than X% percent, then the point is automatically assigned this class
This allows to update the value of the classification for specific points to a value determined by a K-nearest neighbors vote.

```
sml predict config.json unclassified.las ./output/model/ne60_mdNone.pkl --filename classified.las --regularize True -k 30
```

In this example, <code>regularization</code> is enabled and the number of neighbors to use is 30.

## Requirements on the Point Cloud

- The training data should be classified with scalar field named <code>classification</code>
- The labeled and unseen data should contain all the features specified in the <code>configuration</code> file
- Supported formats: <code>.LAS</code> <code>.LAZ</code>

## Citation

If you find our work useful in your research, please consider citing:

```bibtex
@inproceedings{NICE2020,
  title={TESSERAE3D: A BENCHMARK FOR TESSERAE SEMANTIC SEGMENTATION IN 3D POINT CLOUDS},
  author={Kharroubi, A., Van Wersch, L., Billen, R., and Poux, F.},
  booktitle={ISPRS Ann. Photogramm. Remote Sens. Spatial Inf. Sci., V-2-2021, 121–128},
  year={2021}
}
```

## About SemanticML
This software was developped by [Kharroubi Abderrazzaq](https://github.com/akharroubi) and [Anass Yarroudh](https://github.com/Yarroudh), researchers at the Geomatics Unit of the University of Liege. For more detailed information please contact us via <akharroubi@uliege.be> or <ayarroudh@uliege.be>, we are pleased to send you the necessary information.
