package hr.fer.zemris.evolveactivationfunction.nodes;

import hr.fer.zemris.evolveactivationfunction.activationfunction.DerivableNode;
import hr.fer.zemris.genetics.symboregression.IExecutable;
import hr.fer.zemris.genetics.symboregression.IInstantiable;
import hr.fer.zemris.genetics.symboregression.TreeNode;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.Cos;
import org.nd4j.linalg.api.ops.impl.transforms.RectifedLinear;
import org.nd4j.linalg.api.ops.impl.transforms.Sin;
import org.nd4j.linalg.api.ops.impl.transforms.Step;
import org.nd4j.linalg.factory.Nd4j;

public class SinNode extends DerivableNode {
    public static final String NAME = "sin";

    public SinNode() {
        super(NAME, 1);
    }

    @Override
    protected IExecutable<INDArray, INDArray> getExecutable() {
        return (input, node) -> {
            input = ((DerivableNode) node.getChild(0)).execute(input);
            Nd4j.getExecutioner().execAndReturn(new Sin(input));
            return input;
        };
    }

    @Override
    public IDerivable getDerivable() {
        return (input, node) -> {
            INDArray dLdz = ((DerivableNode) node.getChild(0)).derivate(input);
            input = ((DerivableNode) node.getChild(0)).execute(input);
            INDArray out = Nd4j.getExecutioner().execAndReturn(new Cos(input));
            return out.muli(dLdz);
        };
    }

    @Override
    protected IInstantiable<TreeNode<INDArray, INDArray>> getInstantiable() {
        return SinNode::new;
    }
}
