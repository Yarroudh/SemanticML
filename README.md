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
