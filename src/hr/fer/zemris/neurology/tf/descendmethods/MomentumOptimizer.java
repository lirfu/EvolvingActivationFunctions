package hr.fer.zemris.neurology.tf.descendmethods;

import com.sun.istack.NotNull;
import hr.fer.zemris.neurology.tf.TFContext;
import org.tensorflow.Operand;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Variable;

public class MomentumOptimizer<T extends Number> implements IOptimizer<T> {
    private Operand<T> learning_rate_;
    private Operand<T> momentum_;

    public MomentumOptimizer(Operand<T> learning_rate, Operand<T> momentum) {
        learning_rate_ = learning_rate;
        momentum_ = momentum;
    }

    @Override
    public Operand<T> apply(@NotNull Ops tf, @NotNull TFContext context, @NotNull Operand<T> variable, @NotNull Operand<T> gradient, @NotNull Class<T> type) {
        Variable<T> accumulator_ = IOptimizer.getAndInitVariable(tf, context, variable.asOutput().shape(), type);
        return tf.applyMomentum(variable, accumulator_, learning_rate_, gradient, momentum_);
    }
}
