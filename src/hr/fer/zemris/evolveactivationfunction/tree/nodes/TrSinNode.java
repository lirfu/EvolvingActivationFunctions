package hr.fer.zemris.evolveactivationfunction.tree.nodes;

import hr.fer.zemris.genetics.symboregression.IExecutable;
import hr.fer.zemris.genetics.symboregression.IInstantiable;
import hr.fer.zemris.genetics.symboregression.TreeNode;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DoubleBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.Cos;
import org.nd4j.linalg.api.ops.impl.transforms.Sin;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.conditions.AbsValueLessThan;
import org.nd4j.linalg.indexing.conditions.Condition;
import org.nd4j.linalg.indexing.conditions.GreaterThanOrEqual;
import org.nd4j.linalg.indexing.conditions.LessThanOrEqual;
import org.nd4j.linalg.ops.transforms.Transforms;

public class TrSinNode extends DerivableNode {
    public static final String NAME = "trsin";
    private static final double PIH = Math.PI / 2.;

    public TrSinNode() {
        super(NAME, 1);
    }

    @Override
    protected IExecutable<INDArray, INDArray> getExecutable() {
        return (input, node) -> {
            input = ((DerivableNode) node.getChild(0)).execute(input);

            INDArray out = Transforms.max(Transforms.min(input, PIH), -PIH);
            Nd4j.getExecutioner().execAndReturn(new Sin(out));
            return out;
        };
    }

    @Override
    public IDerivable getDerivable() {
        return (input, node) -> {
            INDArray dLdz = ((DerivableNode) node.getChild(0)).derivate(input);
            input = ((DerivableNode) node.getChild(0)).execute(input);

            INDArray out = Transforms.max(Transforms.min(input, PIH), -PIH);
            Nd4j.getExecutioner().execAndReturn(new Cos(out));
            return out.muli(dLdz);
        };
    }

    @Override
    protected IInstantiable<TreeNode<INDArray, INDArray>> getInstantiable() {
        return TrSinNode::new;
    }
}
