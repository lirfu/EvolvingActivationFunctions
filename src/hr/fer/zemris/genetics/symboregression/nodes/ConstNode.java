package hr.fer.zemris.genetics.symboregression.nodes;

import hr.fer.zemris.genetics.symboregression.IExecutable;
import hr.fer.zemris.genetics.symboregression.IInstantiable;
import hr.fer.zemris.genetics.symboregression.TreeNode;

public class ConstNode<I> extends TreeNode<I, Double> {
    public static final String NAME = "const";

    private ConstNode() {
        super(NAME, 0);
    }

    public ConstNode(double init_val) {
        super(NAME, 0, init_val);
    }

    @Override
    protected IInstantiable<TreeNode<I, Double>> getInstantiable() {
        return ConstNode::new;
    }

    @Override
    protected IExecutable<I, Double> getExecutable() {
        return (inp, node) -> (Double) node.getExtra();
    }
}
