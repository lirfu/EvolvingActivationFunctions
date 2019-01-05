package hr.fer.zemris.evolveactivationfunction.nodes;

import hr.fer.zemris.evolveactivationfunction.activationfunction.DerivableNode;
import hr.fer.zemris.genetics.symboregression.IExecutable;
import hr.fer.zemris.genetics.symboregression.IInstantiable;
import hr.fer.zemris.genetics.symboregression.TreeNode;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.Exp;
import org.nd4j.linalg.factory.Nd4j;

public class GaussNode extends DerivableNode {
    public GaussNode() {
        super("gauss", 1);
    }

    @Override
    protected IExecutable<INDArray, INDArray> getExecutable() {
        return (input, node) -> {
            input = ((DerivableNode) node.getChild(0)).execute(input);
            Nd4j.getExecutioner().execAndReturn(new Exp(input.muli(input).muli(-1.)));
            return input;
        };
    }

    @Override
    public IDerivable getDerivable() {
        return (input, node) -> {
            INDArray dLdz = ((DerivableNode) node.getChild(0)).derivate(input);
            INDArray out = Nd4j.getExecutioner().execAndReturn(new Exp(input.mul(input).muli(-1.)));
            return out.muli(-2.).muli(dLdz);
        };
    }

    @Override
    protected IInstantiable<TreeNode<INDArray, INDArray>> getInstantiable() {
        return GaussNode::new;
    }
}
