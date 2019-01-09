package hr.fer.zemris.evolveactivationfunction.nodes;

import hr.fer.zemris.evolveactivationfunction.activationfunction.DerivableNode;
import hr.fer.zemris.genetics.symboregression.IExecutable;
import hr.fer.zemris.genetics.symboregression.IInstantiable;
import hr.fer.zemris.genetics.symboregression.TreeNode;
import org.deeplearning4j.nn.layers.feedforward.PReLU;
import org.nd4j.linalg.activations.impl.ActivationPReLU;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Terminal representing the functions' input.
 */
public class LearnableNode extends DerivableNode {
    private INDArray alpha;

    public LearnableNode() {
        this(0.);
    }

    public LearnableNode(double init_value) {
        super("l", 0, init_value);
    }

    @Override
    protected IExecutable<INDArray, INDArray> getExecutable() {
        return (input, node) -> {//TODO
            DynamicCustomOp.DynamicCustomOpsBuilder prelu = DynamicCustomOp.builder("prelu").addOutputs(input).addInputs(input, alpha);
            Nd4j.getExecutioner().exec(prelu.build());
            return input;
        };
    }

    @Override
    public IDerivable getDerivable() {
        return (input, node) -> Nd4j.zerosLike(input);//TODO

    }

    @Override
    protected IInstantiable<TreeNode<INDArray, INDArray>> getInstantiable() {
        return LearnableNode::new;
    }
}
