package hr.fer.zemris.data.primitives;

import hr.fer.zemris.utils.Pair;
import org.tensorflow.Tensors;

/**
 * Primitive for a batch of input-label pairs (matrix, vector).
 */
public class BatchPair extends Pair<float[][], float[]> implements ITensorablePair<Float> {
    public BatchPair(float[][] key, float[] val) {
        super(key, val);
    }

    @Override
    public TensorPair<Float> tensorify() {
        return new TensorPair<>(Tensors.create(getKey()), Tensors.create(getVal()));
    }
}
