package hr.fer.zemris.evolveactivationfunction.activationfunction;

import hr.fer.zemris.genetics.symboregression.SymbolicTree;
import hr.fer.zemris.genetics.symboregression.TreeNode;
import hr.fer.zemris.genetics.symboregression.TreeNodeSet;
import hr.fer.zemris.neurology.dl4j.ModelReport;
import hr.fer.zemris.utils.Pair;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class DerivableSymbolicTree extends SymbolicTree<INDArray, INDArray> {
    private Pair<ModelReport, INDArray> result_;

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

    @Override
    public DerivableSymbolicTree copy() {
        DerivableSymbolicTree t = new DerivableSymbolicTree(super.copy());
        t.result_ = result_;
        return t;
    }

    public Pair<ModelReport, INDArray> getResult() {
        return result_;
    }

    public void setResult(Pair<ModelReport, INDArray> result) {
        result_ = result;
    }
}
