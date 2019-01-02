package hr.fer.zemris.evolveactivationfunction.activationfunction;

import hr.fer.zemris.genetics.symboregression.TreeNode;
import org.jetbrains.annotations.NotNull;
import org.nd4j.linalg.api.ndarray.INDArray;

public abstract class DerivableNode extends TreeNode<INDArray, INDArray> {
    private IDerivable derivable_;

    public interface IDerivable {
        public INDArray derivate(INDArray input, TreeNode<INDArray, INDArray> node);
    }

    protected DerivableNode(String name, int children_num) {
        super(name, children_num);
    }

    public abstract IDerivable getDerivable();

    public INDArray derivate(INDArray input) {
        return getDerivable().derivate(input, this);
    }

    @Override
    public void swapNodeWith(@NotNull TreeNode n) {
        super.swapNodeWith(n);

        IDerivable der = derivable_;
        derivable_ = ((DerivableNode) n).derivable_;
        ((DerivableNode) n).derivable_ = der;
    }

    @Override
    public void swapContentWith(@NotNull TreeNode n) {
        super.swapContentWith(n);

        IDerivable der = derivable_;
        derivable_ = ((DerivableNode) n).derivable_;
        ((DerivableNode) n).derivable_ = der;
    }
}
