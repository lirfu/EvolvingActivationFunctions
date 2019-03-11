package hr.fer.zemris.evolveactivationfunction.activationfunction;

import hr.fer.zemris.genetics.symboregression.TreeNode;
import org.jetbrains.annotations.NotNull;
import org.nd4j.linalg.api.ndarray.INDArray;

public abstract class DerivableNode extends TreeNode<INDArray, INDArray> {
    private IDerivable derivable_;
    private ILDerivable lderivable_;

    public interface IDerivable {
        public INDArray derivate(INDArray input, TreeNode<INDArray, INDArray> node);
    }

    public interface ILDerivable {
        public INDArray derivate(INDArray input, INDArray epsilon, TreeNode<INDArray, INDArray> node);
    }

    protected DerivableNode(String name, int children_num) {
        super(name, children_num);
        derivable_ = getDerivable();
        lderivable_ = getLDerivable();
    }

    protected DerivableNode(String name, int children_num, Object extra) {
        super(name, children_num, extra);
        derivable_ = getDerivable();
        lderivable_ = getLDerivable();
    }

    public abstract IDerivable getDerivable();

    public abstract ILDerivable getLDerivable();

    public INDArray derivate(INDArray input) {
        return derivable_.derivate(input, this);
    }

    public INDArray lderivate(INDArray input, INDArray epsilon) {
        return lderivable_.derivate(input, epsilon, this);
    }

    @Override
    public void swapInternalsWith(@NotNull TreeNode n) {
        super.swapInternalsWith(n);

        IDerivable der = derivable_;
        derivable_ = ((DerivableNode) n).derivable_;
        ((DerivableNode) n).derivable_ = der;

        ILDerivable lder = lderivable_;
        lderivable_ = ((DerivableNode) n).lderivable_;
        ((DerivableNode) n).lderivable_ = lder;
    }

    @Override
    public void swapAllWith(@NotNull TreeNode n) {
        super.swapAllWith(n);

        IDerivable der = derivable_;
        derivable_ = ((DerivableNode) n).derivable_;
        ((DerivableNode) n).derivable_ = der;

        ILDerivable lder = lderivable_;
        lderivable_ = ((DerivableNode) n).lderivable_;
        ((DerivableNode) n).lderivable_ = lder;
    }

    @Override
    public TreeNode clone() {
        DerivableNode n = (DerivableNode) super.clone();
        n.derivable_ = derivable_;
        return n;
    }
}
