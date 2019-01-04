package hr.fer.zemris.evolveactivationfunction.nodes;

import hr.fer.zemris.evolveactivationfunction.activationfunction.DerivableNode;
import hr.fer.zemris.genetics.symboregression.IExecutable;
import hr.fer.zemris.genetics.symboregression.IInstantiable;
import hr.fer.zemris.genetics.symboregression.TreeNode;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Terminal representing the functions' input.
 */
public class InputNode extends DerivableNode {
    public InputNode() {
        super("inp", 0);
    }


    @Override
    protected IExecutable<INDArray, INDArray> getExecutable() {
        return (input, node) -> input.dup();
    }

    @Override
    public IDerivable getDerivable() {
        return (input, node) -> Nd4j.scalar(1.f);
    }

    @Override
    protected IInstantiable<TreeNode<INDArray, INDArray>> getInstantiable() {
        return InputNode::new;
    }
}
