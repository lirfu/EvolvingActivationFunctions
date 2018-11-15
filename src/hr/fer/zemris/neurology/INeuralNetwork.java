package hr.fer.zemris.neurology;

import hr.fer.zemris.data.APipe;
import hr.fer.zemris.data.primitives.TensorPair;
import hr.fer.zemris.tf.TFContext;
import hr.fer.zemris.tf.TFStep;
import org.tensorflow.Graph;
import org.tensorflow.Operand;
import org.tensorflow.Session;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Placeholder;

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
