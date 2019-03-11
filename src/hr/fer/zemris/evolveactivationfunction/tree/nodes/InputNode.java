package hr.fer.zemris.evolveactivationfunction.tree.nodes;

import hr.fer.zemris.genetics.symboregression.IExecutable;
import hr.fer.zemris.genetics.symboregression.IInstantiable;
import hr.fer.zemris.genetics.symboregression.TreeNode;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Terminal representing the functions' input.
 */
public class InputNode extends DerivableNode {
    public static final String NAME = "x";

    public InputNode() {
        super(NAME, 0);
    }


    @Override
    protected IExecutable<INDArray, INDArray> getExecutable() {
        return (input, node) -> input.dup();
    }

    @Override
    public IDerivable getDerivable() {
        return (input, node) -> Nd4j.onesLike(input);
    }

    @Override
    protected IInstantiable<TreeNode<INDArray, INDArray>> getInstantiable() {
        return InputNode::new;
    }
}
