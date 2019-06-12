package hr.fer.zemris.evolveactivationfunction.tree.nodes;

import hr.fer.zemris.genetics.symboregression.IExecutable;
import hr.fer.zemris.genetics.symboregression.IInstantiable;
import hr.fer.zemris.genetics.symboregression.TreeNode;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.RationalTanh;
import org.nd4j.linalg.api.ops.impl.transforms.RectifiedTanh;
import org.nd4j.linalg.api.ops.impl.transforms.gradient.RationalTanhDerivative;
import org.nd4j.linalg.api.ops.impl.transforms.gradient.RectifiedTanhDerivative;
import org.nd4j.linalg.factory.Nd4j;

public class RectifiedTanhNode extends DerivableNode {
    public static final String NAME = "rectanh";

    public RectifiedTanhNode() {
        super(NAME, 1);
    }

    @Override
    protected IExecutable<INDArray, INDArray> getExecutable() {
        return (input, node) -> {
            input = ((DerivableNode) node.getChild(0)).execute(input);
            Nd4j.getExecutioner().execAndReturn(new RectifiedTanh(input));
            return input;
        };
    }

    @Override
    public IDerivable getDerivable() {
        return (input, node) -> {
            INDArray dLdz = ((DerivableNode) node.getChild(0)).derivate(input);
            input = ((DerivableNode) node.getChild(0)).execute(input);
            INDArray out = Nd4j.getExecutioner().execAndReturn(new RectifiedTanhDerivative(input));
            return out.muli(dLdz);
        };
    }

    @Override
    protected IInstantiable<TreeNode<INDArray, INDArray>> getInstantiable() {
        return RectifiedTanhNode::new;
    }
}
