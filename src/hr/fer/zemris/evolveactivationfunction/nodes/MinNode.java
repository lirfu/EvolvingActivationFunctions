package hr.fer.zemris.evolveactivationfunction.nodes;

import hr.fer.zemris.evolveactivationfunction.activationfunction.DerivableNode;
import hr.fer.zemris.genetics.symboregression.IExecutable;
import hr.fer.zemris.genetics.symboregression.IInstantiable;
import hr.fer.zemris.genetics.symboregression.TreeNode;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;

public class MinNode extends DerivableNode {
    public MinNode() {
        super("min", 2);
    }

    @Override
    protected IExecutable<INDArray, INDArray> getExecutable() {
        return (input, node) -> {
            INDArray output1 = ((DerivableNode) node.getChild(0)).execute(input);
            INDArray output2 = ((DerivableNode) node.getChild(1)).execute(input);
            return Transforms.min(output1, output2);
        };
    }

    @Override
    public IDerivable getDerivable() {
        return (input, node) -> {
            INDArray dLdz1 = ((DerivableNode) node.getChild(0)).derivate(input);
            INDArray dLdz2 = ((DerivableNode) node.getChild(1)).derivate(input);
            return dLdz1.add(dLdz2).muli(0.); // FIXME This won't work
        };
    }

    @Override
    protected IInstantiable<TreeNode<INDArray, INDArray>> getInstantiable() {
        return MinNode::new;
    }
}
