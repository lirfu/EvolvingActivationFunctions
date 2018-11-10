package hr.fer.zemris.data;

public abstract class APipe<I, O> {
    /** The parent pipe.
     * Some components will propagate their calls to their parents.*/
    protected APipe<?, I> parent_;

    /** This method fetches an object from the upstream pipeline, modifies it and returns. */
    public abstract O get();

    /**This method propagates the reset signal upstream.
     * Some components will propagate this call to their parents. */
    public abstract void reset();
}
