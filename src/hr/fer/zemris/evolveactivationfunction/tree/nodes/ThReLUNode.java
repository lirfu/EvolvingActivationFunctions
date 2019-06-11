package hr.fer.zemris.evolveactivationfunction.tree.nodes;

import hr.fer.zemris.genetics.symboregression.IExecutable;
import hr.fer.zemris.genetics.symboregression.IInstantiable;
import hr.fer.zemris.genetics.symboregression.TreeNode;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.impl.transforms.RectifedLinear;
import org.nd4j.linalg.api.ops.impl.transforms.Step;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Pair;

public class ThReLUNode extends DerivableNode {
    public static final String NAME = "threlu";
    private final static double theta = 1.;

    public ThReLUNode() {
        super(NAME, 1);
    }

    @Override
    protected IExecutable<INDArray, INDArray> getExecutable() {
        return (input, node) -> {
            input = ((DerivableNode) node.getChild(0)).execute(input);
            DynamicCustomOp threshRelu = DynamicCustomOp.builder("thresholdedrelu")
                    .addOutputs(input).addInputs(input)
                    .addFloatingPointArguments(theta).build();
            Nd4j.getExecutioner().exec(threshRelu);
            return input;
        };
    }

    @Override
    public IDerivable getDerivable() {
        return (input, node) -> {
            INDArray dLdz = ((DerivableNode) node.getChild(0)).derivate(input);
            input = ((DerivableNode) node.getChild(0)).execute(input);

            DynamicCustomOp threshReluBp = DynamicCustomOp.builder("thresholdedrelu_bp")
                    .addInputs(input, dLdz).addOutputs(input).addFloatingPointArguments(theta).build();
            Nd4j.getExecutioner().exec(threshReluBp);
            return input;
        };
    }

    @Override
    protected IInstantiable<TreeNode<INDArray, INDArray>> getInstantiable() {
        return ThReLUNode::new;
    }
}
