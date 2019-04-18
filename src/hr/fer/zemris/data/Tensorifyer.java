package hr.fer.zemris.data;

import com.sun.istack.NotNull;
import hr.fer.zemris.data.primitives.ITensorablePair;
import hr.fer.zemris.data.primitives.TensorPair;

/**
 * Turns the object from the input stream into pairs of tensors.
 */
public class Tensorifyer<T> extends APipe<ITensorablePair<T>, TensorPair<T>> {
    public Tensorifyer(@NotNull APipe<?, ITensorablePair<T>> parent) {
        parent_ = parent;
    }

    /**
     * Gets a single input from parent and returns a tensor representation of it.
     *
     * @return Tensor created from the parent input.
     */
    @Override
    public TensorPair<T> next() {
        ITensorablePair<T> d = parent_.next();
        if (d == null) {
            return null;
        }
        return d.tensorify();
    }

    /**
     * Resets the parent.
     */
    @Override
    public void reset() {
        parent_.reset();
    }

    @Override
    public APipe<ITensorablePair<T>, TensorPair<T>> clone() {
        return new Tensorifyer<>(parent_);
    }
}
