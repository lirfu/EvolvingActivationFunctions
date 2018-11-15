package hr.fer.zemris.neurology.descendmethods;

import com.sun.istack.internal.NotNull;
import hr.fer.zemris.tf.TFContext;
import org.tensorflow.Operand;
import org.tensorflow.Shape;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Assign;
import org.tensorflow.op.core.Variable;

public interface IOptimizer<T extends Number> {
    public Operand<T> apply(@NotNull Ops tf, @NotNull TFContext context, @NotNull Operand<T> variable, @NotNull Operand<T> gradient, @NotNull Class<T> type);

    static Variable getAndInitVariable(@NotNull Ops tf, @NotNull TFContext context, @NotNull Shape shape, @NotNull Class type) {
        Variable var = tf.variable(shape, type);
        Assign acc_init = tf.assign(var, tf.zeros(tf.shape(var), type));
        context.addTarget(acc_init);
        return var;
    }
}
