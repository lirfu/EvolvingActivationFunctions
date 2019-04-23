# About
This project attempts evolving transfer functions for neural networks using symbolic regression.
Project is implemented using [DeepLearning4Java](https://deeplearning4j.org/).

# Installation procedure
The installation procedure can be found [here](/DL4J_setup_procedure.md)

# Main components
## Packages
#### `hr.fer.zemris.evolveactivationfunction`
* Defines common structures and procedures for the evolution process.

#### `hr.fer.zemris.genetics`
* Evolutionary search algorithms and structures. 

#### `hr.fer.zemris.architecturesearch`
* Methods for search and analysis of the initial architecture for a given dataset.

#### `hr.fer.zemris.data`
* Custom data structures and pipelines.
* Includes `Parser` for DPAv2/4 CSV files.

#### `hr.fer.zemris.neurology`
* Demos on the usage of DL4J and TF frameworks.

#### `hr.fer.zemris.utils`
* Various common utilities.

## Structures and algorithms
#### `hr.fer.zemris.evolveactivationfunction.nn.CommonModel`
* Wrapper for the neural network, used to hide unnecessary features
* Softly prevents unwanted model modifications (for comparable results).

#### `hr.fer.zemris.evolveactivationfunction.nn.TrainProcedure`
* Defines the common procedures and structures used for model training.
* Ensures comparability of experiment results.

#### `hr.fer.zemris.neurology.dl4j.TrainParams`
* Defines immutable parameters used for model training.
* Ensures comparability of experiment results.

#### `hr.fer.zemris.neurology.dl4j.ModelReport`
* Defines model performance on the test set.

#### `hr.fer.zemris.evolveactivationfunction.StorageManager`
* Workhorse for procedures related to storing and loading experiments.

#### `hr.fer.zemris.evolveactivationfunction.Context`
* Defines the storage paths of an experiment.

## Programs and demos
#### `hr.fer.zemris.architecturesearch.Main`
* Used to run custom experiments using the DL4J model and data API

#### `hr.fer.zemris.data.DemoTests`
* Demonstrates the usage of the custom data pipeline.

#### `hr.fer.zemris.neurology.DemoDL4J`
* Demonstrates learning a neural network using DeepLearning4Java on MNIST and custom datasets. 

#### `hr.fer.zemris.neurology.DemoTF`
* Demonstrates learning a neural network using Tensorflow Java Ops API.

## Workflow pipeline (deprecated)

#### 1. Loading data
To load data construct a pipeline.
* Parser - reads dataset description markers and passes actual data through
* Cacher - loads the whole stream into an array (for faster inference) and additionally applying some data modifiers 
* Batcher - constructs batches from input stream
* DatumF - wrapper class used to construct and hold a Tensor pair (doesn't matter if batch or not)
* IModifier - interface for constructing modifiers used to modify the data (normalization, randomization before batching, etc.)
  * Randomizer - randomizes the order of instances in dataset

Method `get()` fetches the next data, method `reset()` generally resets internals to start over the inference procedure. 


