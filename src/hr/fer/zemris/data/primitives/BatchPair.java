package hr.fer.zemris.data.primitives;

import hr.fer.zemris.utils.Pair;

/**
 * Primitive for a batch of input-label pairs (matrix, vector).
 */
public class BatchPair extends Pair<float[][], float[]> {
    public BatchPair(float[][] key, float[] val) {
        super(key, val);
    }
}
