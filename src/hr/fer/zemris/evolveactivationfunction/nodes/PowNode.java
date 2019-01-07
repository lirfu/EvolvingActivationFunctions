package hr.fer.zemris.evolveactivationfunction.nodes;

import hr.fer.zemris.evolveactivationfunction.activationfunction.DerivableNode;
import hr.fer.zemris.genetics.symboregression.IExecutable;
import hr.fer.zemris.genetics.symboregression.IInstantiable;
import hr.fer.zemris.genetics.symboregression.TreeNode;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;


public class PowNode extends DerivableNode {
    public PowNode() {
        super("^", 2);
    }

    @Override
    protected IExecutable<INDArray, INDArray> getExecutable() {
        return (input, node) -> {
            INDArray input1 = ((DerivableNode) node.getChild(0)).execute(input);
            INDArray input2 = ((DerivableNode) node.getChild(1)).execute(input);
            return Transforms.pow(input1, input2);
        };
    }

    @Override
    public IDerivable getDerivable() {
        return (input, node) -> {
            INDArray out1 = ((DerivableNode) node.getChild(0)).execute(input);
            INDArray out2 = ((DerivableNode) node.getChild(1)).execute(input);
            INDArray dLdz1 = ((DerivableNode) node.getChild(0)).derivate(input);
            INDArray dLdz2 = ((DerivableNode) node.getChild(1)).derivate(input);
            return Transforms.pow(out1, out2.sub(1.))
                    .muli(out2.muli(dLdz1).addi(
                            out1.muli(Transforms.log(out1)).muli(dLdz2)
                    ));
        };
    }

    @Override
    protected IInstantiable<TreeNode<INDArray, INDArray>> getInstantiable() {
        return PowNode::new;
    }
}
