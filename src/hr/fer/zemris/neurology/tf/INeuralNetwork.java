package hr.fer.zemris.neurology.tf;

import hr.fer.zemris.data.APipe;

public interface INeuralNetwork<T> {
    /**
     * 1st method. Builds the graph and defines all the inputs/outputs.
     */
    public void build();

    /**
     * 2nd method. Initializes the variables and parameters.
     */
    public void initialize();

    /**
     * Trains the parameters using the provided dataset.
     */
    public void train(APipe<?, T> dataset);

    /**
     * Returns predictions of the given inputs.
     */
    public T[] predict(APipe<?, T> dataset);
}
