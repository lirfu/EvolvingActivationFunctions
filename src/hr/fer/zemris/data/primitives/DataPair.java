package hr.fer.zemris.data.primitives;

import hr.fer.zemris.utils.Pair;

/** Primitive for a single input-label pair (vector, scalar). */
public class DataPair extends Pair<float[], Float> {
    public DataPair(float[] key, Float val) {
        super(key, val);
    }
}
