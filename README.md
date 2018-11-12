# Workflow pipeline

## 1. Loading data
To load data construct a pipeline.
* Reader - reads a file line by line
* Parser - reads dataset description markers and passes actual data through
* Cacher - loads the whole stream into an array (for faster inference) and additionally applying some data modifiers 
* Batcher - constructs batches from input stream
* DatumF - wrapper class used to construct and hold a Tensor pair (doesn't matter if batch or not)
* IModifier - interface for constructing modifiers used to modify the data (normalization, randomization before batching, etc.)
  * Randomizer - randomizes the order of instances in dataset

Method `get()` fetches the next data, method `reset()` generally resets internals to start over the inference procedure. 
