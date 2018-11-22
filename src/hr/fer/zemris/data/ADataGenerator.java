package hr.fer.zemris.data;

import hr.fer.zemris.data.primitives.DataPair;


/**
 * Just a common ancestor for all data generators.
 */
public abstract class ADataGenerator<I> extends APipe<I, DataPair> implements IDescriptableDS {
}
