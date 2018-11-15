package hr.fer.zemris.neurology.descendmethods;

import hr.fer.zemris.tf.TFContext;
import org.tensorflow.Operand;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Variable;

public class AdamOptimizer implements IOptimizer<Float> {
    private Operand<Float> learning_rate_;
    private Operand<Float> t_;

    public AdamOptimizer(Operand<Float> learning_rate, Operand<Float> t) {
        learning_rate_ = learning_rate;
        t_ = t;
    }

    @Override
    public Operand<Float> apply(Ops tf, TFContext context, Operand<Float> variable, Operand<Float> gradient, Class<Float> type) {
        Variable<Float> m = IOptimizer.getAndInitVariable(tf, context, variable.asOutput().shape(), type);
        Variable<Float> v = IOptimizer.getAndInitVariable(tf, context, variable.asOutput().shape(), type);
        return tf.applyAdam(variable, m, v, t_, t_,
                learning_rate_,
                tf.constant(0.9f), tf.constant(0.999f), tf.constant(1e-8f),
                gradient);
    }
}
