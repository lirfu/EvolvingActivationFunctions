package hr.fer.zemris.data.primitives;


import hr.fer.zemris.utils.Pair;
import org.tensorflow.Tensor;

public class TensorPair<T> extends Pair<Tensor<T>, Tensor<T>> implements ITensorablePair<T> {
    public TensorPair(Tensor<T> key, Tensor<T> val) {
        super(key, val);
    }

    /**
     * Returns itself.
     */
    @Override
    public TensorPair<T> tensorify() {
        return this;
    }
}
