package hr.fer.zemris.evolveactivationfunction.nodes;

import hr.fer.zemris.evolveactivationfunction.activationfunction.DerivableNode;
import hr.fer.zemris.genetics.symboregression.IExecutable;
import hr.fer.zemris.genetics.symboregression.IInstantiable;
import hr.fer.zemris.genetics.symboregression.TreeNode;
import org.nd4j.linalg.activations.impl.ActivationPReLU;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Terminal representing the functions' input.
 */
public class LearnableNode extends DerivableNode {
    public static final String NAME = "l";

    private INDArray const_input = Nd4j.scalar(-1.f);
    private INDArray alpha;
    private ActivationPReLU prelu;

    public LearnableNode() {
        this(1.);
    }

    public LearnableNode(double init_value) {
        super(NAME, 0);
        alpha = Nd4j.scalar(1.);
        super.extra_ = alpha;  // Set the parameter as extra.
        prelu = new ActivationPReLU(alpha);
    }

    public INDArray getAlpha() {
        return alpha;
    }

    @Override
    protected IExecutable<INDArray, INDArray> getExecutable() {
        return (input, node) -> prelu.getActivation(const_input, false);
    }

    @Override
    public IDerivable getDerivable() {
        return (input, node) -> Nd4j.scalar(0.); // Not dependent on input.

    }

    @Override
    public ILDerivable getLDerivable() {
        return (input, epsilon, node) -> prelu.backprop(const_input, epsilon).getFirst(); // TODO Does this update alpha? Check in org.deeplearning4j.nn.layers.feedforward.PReLU
    }

    @Override
    protected IInstantiable<TreeNode<INDArray, INDArray>> getInstantiable() {
        return LearnableNode::new;
    }
}
