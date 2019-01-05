package hr.fer.zemris.evolveactivationfunction.nodes;

import hr.fer.zemris.evolveactivationfunction.activationfunction.DerivableNode;
import hr.fer.zemris.genetics.symboregression.IExecutable;
import hr.fer.zemris.genetics.symboregression.IInstantiable;
import hr.fer.zemris.genetics.symboregression.TreeNode;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.Cube;
import org.nd4j.linalg.api.ops.impl.transforms.Pow;
import org.nd4j.linalg.api.ops.impl.transforms.PowDerivative;
import org.nd4j.linalg.api.ops.impl.transforms.gradient.CubeDerivative;
import org.nd4j.linalg.factory.Nd4j;

public class Pow3Node extends DerivableNode {
    public Pow3Node() {
        super("^3", 1);
    }

    @Override
    protected IExecutable<INDArray, INDArray> getExecutable() {
        return (input, node) -> {
            input = ((DerivableNode) node.getChild(0)).execute(input);
            Nd4j.getExecutioner().execAndReturn(new Cube(input));
            return input;
        };
    }

    @Override
    public IDerivable getDerivable() {
        return (input, node) -> {
            INDArray dLdz = ((DerivableNode) node.getChild(0)).derivate(input);
            INDArray out = Nd4j.getExecutioner().execAndReturn(new CubeDerivative(input.dup()));
            return out.muli(dLdz);
        };
    }

    @Override
    protected IInstantiable<TreeNode<INDArray, INDArray>> getInstantiable() {
        return Pow3Node::new;
    }
}
