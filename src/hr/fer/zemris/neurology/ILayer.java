package hr.fer.zemris.neurology;

import com.sun.istack.internal.NotNull;
import hr.fer.zemris.tf.TFContext;
import org.tensorflow.Operand;

import java.util.List;

public interface ILayer<T> {
    /**
     * Builds the layers' graph on top of the given input operand.
     *
     * @param input Input of the layer.
     * @return Output of the layer.
     */
    public Operand<T> build(@NotNull Operand<T> input);

    /**
     * Initializes layers' weights and bias with given seed.
     *
     * @param context Context for initialization.
     * @param seed    Seed for the random initializer.
     */
    public void initialize(@NotNull TFContext context, long seed);

    /**
     * Registers layers' operands that get updated during training.
     *
     * @param gradients Container for gradients.
     */
    public void registerGradients(@NotNull List<Operand<T>> gradients);

    /**
     * Registers parameters that can get fetched from the given context.
     *
     * @param context Context that will fetch layers' parameters.
     */
    public void registerTargetToFetch(@NotNull TFContext context);
}
