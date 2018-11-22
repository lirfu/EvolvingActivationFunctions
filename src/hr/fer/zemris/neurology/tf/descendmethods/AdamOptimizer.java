package hr.fer.zemris.neurology.tf.descendmethods;

import hr.fer.zemris.neurology.tf.TFContext;
import org.tensorflow.Operand;
import org.tensorflow.Shape;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Variable;

public class AdamOptimizer implements IOptimizer<Float> {
    private Operand<Float> learning_rate_;

    public AdamOptimizer(Operand<Float> learning_rate) {
        learning_rate_ = learning_rate;
    }

    @Override
    public Operand<Float> apply(Ops tf, TFContext context, Operand<Float> variable, Operand<Float> gradient, Class<Float> type) {
        Variable<Float> grad_acc = IOptimizer.getAndInitVariable(tf, context, variable.asOutput().shape(), type);
        Variable<Float> grad_sq_acc = IOptimizer.getAndInitVariable(tf, context, variable.asOutput().shape(), type);
        Variable<Float> t = IOptimizer.getAndInitVariable(tf, context, Shape.scalar(), type);
        return tf.applyAdam(variable, grad_acc, grad_sq_acc, t, t,
                learning_rate_,
                tf.constant(0.9f), tf.constant(0.999f), tf.constant(1e-8f),
                gradient);
    }
}
