package hr.fer.zemris.evolveactivationfunction.tree.nodes;

import hr.fer.zemris.genetics.symboregression.IExecutable;
import hr.fer.zemris.genetics.symboregression.IInstantiable;
import hr.fer.zemris.genetics.symboregression.TreeNode;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.Cos;
import org.nd4j.linalg.api.ops.impl.transforms.Sin;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.conditions.AbsValueLessThan;
import org.nd4j.linalg.ops.transforms.Transforms;

public class PLUNode extends DerivableNode {
    public static final String NAME = "plu";
    private static double alpha, c; // FIXME Can't have non-final internal attributes because node instances are shared across trees.

    public PLUNode() {
        super(NAME, 1);
        alpha = 0.1;
        c = 1;
    }

    public PLUNode(double alpha, double c) {
        super(NAME, 1);
        this.alpha = alpha;
        this.c = c;
    }

    @Override
    protected IExecutable<INDArray, INDArray> getExecutable() {
        return (input, node) -> {
            input = ((DerivableNode) node.getChild(0)).execute(input);

            INDArray out = Transforms.max(input.mul(alpha).subi(c * (1 - alpha)),
                    Transforms.min(input, input.mul(alpha).addi(c * (1 - alpha))));
            return out;
        };
    }

    @Override
    public IDerivable getDerivable() {
        return (input, node) -> {
            INDArray dLdz = ((DerivableNode) node.getChild(0)).derivate(input);
            input = ((DerivableNode) node.getChild(0)).execute(input);

            INDArray out = input.condi(new AbsValueLessThan(c));
//            INDArray out = Transforms.max(Transforms.min(input, c), -c);
            out = out.add(alpha).subi(out.mul(alpha));
            return out.muli(dLdz);
        };
    }

    @Override
    protected IInstantiable<TreeNode<INDArray, INDArray>> getInstantiable() {
        return PLUNode::new;
    }
}
