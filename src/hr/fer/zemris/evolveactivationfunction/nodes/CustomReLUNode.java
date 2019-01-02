package hr.fer.zemris.evolveactivationfunction.nodes;

import hr.fer.zemris.evolveactivationfunction.activationfunction.DerivableNode;
import hr.fer.zemris.genetics.symboregression.IExecutable;
import hr.fer.zemris.genetics.symboregression.IInstantiable;
import hr.fer.zemris.genetics.symboregression.TreeNode;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.scalar.ScalarMax;
import org.nd4j.linalg.api.ops.impl.transforms.RectifedLinear;
import org.nd4j.linalg.api.ops.impl.transforms.Step;
import org.nd4j.linalg.factory.Nd4j;

public class CustomReLUNode extends DerivableNode {
    public CustomReLUNode() {
        super("c_relu", 1);
    }

    @Override
    protected IExecutable<INDArray, INDArray> getExecutable() {
        return (input, node) -> {
            input = ((DerivableNode) node.getChild(0)).execute(input);
            Nd4j.getExecutioner().execAndReturn(new ScalarMax(input, 0.));
            return input;
        };
    }

    @Override
    public IDerivable getDerivable() {
        return (input, node) -> {
            INDArray dLdz = ((DerivableNode) node.getChild(0)).derivate(input);
            Nd4j.getExecutioner().execAndReturn(new Step(input));
            input.mul(dLdz);
            return input;
        };
    }

    @Override
    protected IInstantiable<TreeNode<INDArray, INDArray>> getInstantiable() {
        return CustomReLUNode::new;
    }
}
