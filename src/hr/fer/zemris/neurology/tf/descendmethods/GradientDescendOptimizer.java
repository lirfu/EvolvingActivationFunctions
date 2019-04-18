package hr.fer.zemris.neurology.tf.descendmethods;

import com.sun.istack.NotNull;
import hr.fer.zemris.neurology.tf.TFContext;
import org.tensorflow.Operand;
import org.tensorflow.op.Ops;

public class GradientDescendOptimizer<T extends Number> implements IOptimizer<T> {
    private Operand<T> learning_rate_;

    public GradientDescendOptimizer(@NotNull Operand<T> learning_rate) {
        learning_rate_ = learning_rate;
    }

    @Override
    public Operand<T> apply(@NotNull Ops tf, @NotNull TFContext context, @NotNull Operand<T> variable, @NotNull Operand<T> gradient, @NotNull Class<T> type) {
        return tf.applyGradientDescent(variable, learning_rate_, gradient);
    }
}
