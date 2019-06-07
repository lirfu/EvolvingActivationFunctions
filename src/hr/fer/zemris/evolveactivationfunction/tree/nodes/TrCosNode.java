package hr.fer.zemris.evolveactivationfunction.tree.nodes;

import hr.fer.zemris.genetics.symboregression.IExecutable;
import hr.fer.zemris.genetics.symboregression.IInstantiable;
import hr.fer.zemris.genetics.symboregression.TreeNode;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.Cos;
import org.nd4j.linalg.api.ops.impl.transforms.Sin;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.conditions.AbsValueLessThan;
import org.nd4j.linalg.indexing.conditions.GreaterThanOrEqual;
import org.nd4j.linalg.indexing.conditions.LessThanOrEqual;
import org.nd4j.linalg.ops.transforms.Transforms;

public class TrCosNode extends DerivableNode {
    public static final String NAME = "trcos";
    private static final double PIH = Math.PI / 2.;

    public TrCosNode() {
        super(NAME, 1);
    }

    @Override
    protected IExecutable<INDArray, INDArray> getExecutable() {
        return (input, node) -> {
            input = ((DerivableNode) node.getChild(0)).execute(input);

            INDArray out = Transforms.min(Transforms.max(input, PIH), -PIH);
            Nd4j.getExecutioner().execAndReturn(new Cos(out));
            return out;
        };
    }

    @Override
    public IDerivable getDerivable() {
        return (input, node) -> {
            INDArray dLdz = ((DerivableNode) node.getChild(0)).derivate(input);
            input = ((DerivableNode) node.getChild(0)).execute(input);

            INDArray out = Transforms.min(Transforms.max(input, PIH), -PIH);
            Nd4j.getExecutioner().execAndReturn(new Sin(out));
            return out.muli(-1.).muli(dLdz);
        };
    }

    @Override
    protected IInstantiable<TreeNode<INDArray, INDArray>> getInstantiable() {
        return TrCosNode::new;
    }
}
