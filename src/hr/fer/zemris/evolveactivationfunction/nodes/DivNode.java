package hr.fer.zemris.evolveactivationfunction.nodes;

import hr.fer.zemris.evolveactivationfunction.activationfunction.DerivableNode;
import hr.fer.zemris.genetics.symboregression.IExecutable;
import hr.fer.zemris.genetics.symboregression.IInstantiable;
import hr.fer.zemris.genetics.symboregression.TreeNode;
import org.nd4j.linalg.api.ndarray.INDArray;

public class DivNode extends DerivableNode {
    private static double stability_const_ = 1e-12;

    public DivNode() {
        super("/", 2);
    }

    @Override
    protected IExecutable<INDArray, INDArray> getExecutable() {
        return (input, node) -> {
            INDArray output1 = ((DerivableNode) node.getChild(0)).execute(input);
            INDArray output2 = ((DerivableNode) node.getChild(1)).execute(input);
            return output1.divi(output2.addi(stability_const_));
        };
    }

    @Override
    public IDerivable getDerivable() {
        return (input, node) -> {
            INDArray output1 = ((DerivableNode) node.getChild(0)).execute(input);
            INDArray output2 = ((DerivableNode) node.getChild(1)).execute(input);
            INDArray dLdz1 = ((DerivableNode) node.getChild(0)).derivate(input);
            INDArray dLdz2 = ((DerivableNode) node.getChild(1)).derivate(input);
            return dLdz1.muli(output2).subi(output1.muli(dLdz2)).divi(output2.muli(output2.add(stability_const_)));
        };
    }

    @Override
    protected IInstantiable<TreeNode<INDArray, INDArray>> getInstantiable() {
        return DivNode::new;
    }
}
