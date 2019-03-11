package hr.fer.zemris.evolveactivationfunction.tree.nodes;

import hr.fer.zemris.genetics.symboregression.IExecutable;
import hr.fer.zemris.genetics.symboregression.IInstantiable;
import hr.fer.zemris.genetics.symboregression.TreeNode;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Terminal representing the functions' input.
 */
public class ConstNode extends DerivableNode {
    public static final String NAME = hr.fer.zemris.genetics.symboregression.nodes.ConstNode.NAME;

    public ConstNode() {
        this(1.);
    }

    public ConstNode(double init_value) {
        super(NAME, 0, init_value);
    }

    @Override
    protected IExecutable<INDArray, INDArray> getExecutable() {
        return (input, node) -> Nd4j.onesLike(input).muli((Double) node.getExtra());
    }

    @Override
    public IDerivable getDerivable() {
        return (input, node) -> Nd4j.zerosLike(input);
    }

    @Override
    protected IInstantiable<TreeNode<INDArray, INDArray>> getInstantiable() {
        return ConstNode::new;
    }
}
