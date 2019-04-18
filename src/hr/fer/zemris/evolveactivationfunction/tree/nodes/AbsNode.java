package hr.fer.zemris.evolveactivationfunction.tree.nodes;

import hr.fer.zemris.genetics.symboregression.IExecutable;
import hr.fer.zemris.genetics.symboregression.IInstantiable;
import hr.fer.zemris.genetics.symboregression.TreeNode;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.Abs;
import org.nd4j.linalg.api.ops.impl.transforms.Cube;
import org.nd4j.linalg.api.ops.impl.transforms.gradient.CubeDerivative;
import org.nd4j.linalg.factory.Nd4j;

public class AbsNode extends DerivableNode {
    public static final String NAME = "abs";
    public static double STABILITY_CONST = 1e-12;

    public AbsNode() {
        super(NAME, 1);
    }

    @Override
    protected IExecutable<INDArray, INDArray> getExecutable() {
        return (input, node) -> {
            input = ((DerivableNode) node.getChild(0)).execute(input);
            Nd4j.getExecutioner().execAndReturn(new Abs(input));
            return input;
        };
    }

    @Override
    public IDerivable getDerivable() {
        return (input, node) -> {
            INDArray dLdz = ((DerivableNode) node.getChild(0)).derivate(input);
            input = ((DerivableNode) node.getChild(0)).execute(input);
            INDArray out = Nd4j.getExecutioner().execAndReturn(new Abs(input.dup()));
            return out.divi(input.add(STABILITY_CONST)).muli(dLdz);
        };
    }

    @Override
    protected IInstantiable<TreeNode<INDArray, INDArray>> getInstantiable() {
        return AbsNode::new;
    }
}
