package hr.fer.zemris.evolveactivationfunction.activationfunction;

import hr.fer.zemris.genetics.symboregression.SymbolicTree;
import hr.fer.zemris.genetics.symboregression.TreeNode;
import hr.fer.zemris.genetics.symboregression.TreeNodeSet;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.primitives.Pair;

public class DerivableSymbolicTree extends SymbolicTree<INDArray, INDArray> {
    private DerivableSymbolicTree(SymbolicTree<INDArray, INDArray> tree) {
        super(tree);
    }

    public DerivableSymbolicTree(TreeNodeSet factory, TreeNode root) {
        super(factory, root);
    }

    public INDArray derivate(INDArray input) {
        return ((DerivableNode) root_).derivate(input);
    }

    public static class Builder extends SymbolicTree.Builder {
        @Override
        public DerivableSymbolicTree build() {
            return new DerivableSymbolicTree(super.build());
        }
    }
}
