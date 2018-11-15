package hr.fer.zemris.neurology;

import com.sun.istack.internal.NotNull;
import com.sun.istack.internal.Nullable;
import hr.fer.zemris.tf.TFContext;
import org.tensorflow.Operand;
import org.tensorflow.Shape;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Assign;
import org.tensorflow.op.core.RandomUniform;
import org.tensorflow.op.core.Variable;

import java.util.List;

/**
 * Defines a fully-connected layer.
 */
public class FullyConnectedLayer<T extends Number> implements ILayer<T> {
    private Ops tf;
    private int size_;
    private IActivationFunction<T> activationFunction_;
    private Class<T> classtype_;

    private Variable<T> w;
    private Variable<T> b;
    private Operand<T> o;

    /**
     * Defines a fully-connected layer.
     *
     * @param ops                Ops object for constructing the TF graph.
     * @param size               Number of neurons.
     * @param activationFunction Activation for the layer (ignored if null).
     * @param classtype          Class instance of the specified type (needed for TF functions).
     */
    public FullyConnectedLayer(@NotNull Ops ops, int size, @Nullable IActivationFunction<T> activationFunction, @NotNull Class<T> classtype) {
        tf = ops;
        size_ = size;
        activationFunction_ = activationFunction;
        classtype_ = classtype;
    }

    public Operand<T> build(@NotNull Operand<T> input) {
        Shape sh = input.asOutput().shape();
        w = tf.variable(Shape.make(sh.size(sh.numDimensions() - 1), size_), classtype_);
        b = tf.variable(Shape.make(size_), classtype_);
        // Calculate output.
        o = tf.add(tf.matMul(input, w), b);
        // Activate output.
        if (activationFunction_ == null)
            return o;
        else
            return activationFunction_.buildUpon(o);
    }

    public void initialize(@NotNull TFContext context, long seed) {
        Assign<T> w_init = tf.assign(w, tf.randomUniform(tf.shape(w), classtype_, RandomUniform.seed(seed)));
        Assign<T> b_init = tf.assign(b, tf.zeros(tf.constant(new long[]{size_}), classtype_));
        context.addTarget(w_init);
        context.addTarget(b_init);
    }

    public void registerGradients(@NotNull List<Operand<T>> gradients) {
        gradients.add(w);
        gradients.add(b);
    }

    public void registerTargetToFetch(@NotNull TFContext context) {
        context.addTargetToFetch(b);
        context.addTargetToFetch(w);
        context.addTargetToFetch(o);
    }

    public Variable<T> getWeigths() {
        return w;
    }

    public Variable<T> getBiases() {
        return b;
    }

    public Operand<T> getOutput() {
        return o;
    }
}
