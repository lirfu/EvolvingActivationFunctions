package hr.fer.zemris.evolveactivationfunction.tree.nodes;

import hr.fer.zemris.genetics.symboregression.IExecutable;
import hr.fer.zemris.genetics.symboregression.IInstantiable;
import hr.fer.zemris.genetics.symboregression.TreeNode;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.Step;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

public class MinNode extends DerivableNode {
    public static final String NAME = "min";

    public MinNode() {
        super(NAME, 2);
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
            INDArray output1 = ((DerivableNode) node.getChild(0)).execute(input);
            INDArray output2 = ((DerivableNode) node.getChild(1)).execute(input);

            INDArray cond = output1.sub(output2);
            INDArray mask1 = Nd4j.getExecutioner().execAndReturn(new Step(cond.mul(-1.)));
            INDArray mask2 = Nd4j.getExecutioner().execAndReturn(new Step(cond));

            INDArray dLdz1 = ((DerivableNode) node.getChild(0)).derivate(input);
            INDArray dLdz2 = ((DerivableNode) node.getChild(1)).derivate(input);
            return dLdz1.muli(mask1).addi(dLdz2.muli(mask2));
        };
    }

    @Override
    protected IInstantiable<TreeNode<INDArray, INDArray>> getInstantiable() {
        return MinNode::new;
    }
}
