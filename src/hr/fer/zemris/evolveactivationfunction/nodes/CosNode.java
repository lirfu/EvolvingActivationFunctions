package hr.fer.zemris.evolveactivationfunction.nodes;

import hr.fer.zemris.evolveactivationfunction.activationfunction.DerivableNode;
import hr.fer.zemris.genetics.symboregression.IExecutable;
import hr.fer.zemris.genetics.symboregression.IInstantiable;
import hr.fer.zemris.genetics.symboregression.TreeNode;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.Cos;
import org.nd4j.linalg.api.ops.impl.transforms.Sin;
import org.nd4j.linalg.factory.Nd4j;

public class CosNode extends DerivableNode {
    public CosNode() {
        super("cos", 1);
    }

    @Override
    protected IExecutable<INDArray, INDArray> getExecutable() {
        return (input, node) -> {
            input = ((DerivableNode) node.getChild(0)).execute(input);
            Nd4j.getExecutioner().execAndReturn(new Cos(input));
            return input;
        };
    }

    @Override
    public IDerivable getDerivable() {
        return (input, node) -> {
            INDArray dLdz = ((DerivableNode) node.getChild(0)).derivate(input);
            INDArray out = Nd4j.getExecutioner().execAndReturn(new Sin(input.dup()));
            return out.muli(-1.).muli(dLdz);
        };
    }

    @Override
    protected IInstantiable<TreeNode<INDArray, INDArray>> getInstantiable() {
        return CosNode::new;
    }
}
