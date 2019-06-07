package hr.fer.zemris.evolveactivationfunction.tree.nodes;

import hr.fer.zemris.genetics.symboregression.IExecutable;
import hr.fer.zemris.genetics.symboregression.IInstantiable;
import hr.fer.zemris.genetics.symboregression.TreeNode;
import hr.fer.zemris.neurology.tf.descendmethods.AdamOptimizer;
import org.nd4j.linalg.activations.impl.ActivationPReLU;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;

/**
 * Terminal representing the functions' input.
 */
public class LearnableNode extends DerivableNode {
    public static final String NAME = "l";
    private ActivationPReLU prelu;

    public LearnableNode() {
        this(0.);
    }

    public LearnableNode(double init_value) {
        super(NAME, 0, init_value);
        prelu = new ActivationPReLU(Nd4j.scalar(1.));
    }

    @Override
    protected IExecutable<INDArray, INDArray> getExecutable() {
        return (input, node) -> prelu.getActivation(Nd4j.scalar(-1), false).muli(-1); // TODO When am I training (bool)??
    }

    @Override
    public IDerivable getDerivable() {
        return (input, node) -> {
            prelu.backprop(input, null);// TODO Get gradient (epsilon) and transform it towards here.... (way too much work)
            return Nd4j.scalar(0);
        };

    }

    @Override
    protected IInstantiable<TreeNode<INDArray, INDArray>> getInstantiable() {
        return LearnableNode::new;
    }
}
