package hr.fer.zemris.data.primitives;


public interface ITensorablePair<T> {
    /**
     * Turns this into a tensor pair.
     */
    public TensorPair<T> tensorify();
}
