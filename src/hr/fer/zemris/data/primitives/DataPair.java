package hr.fer.zemris.data.primitives;

import hr.fer.zemris.utils.Pair;
import org.tensorflow.Tensors;

/**
 * Primitive for a single input-label pair (vector, scalar).
 */
public class DataPair extends Pair<float[], float[]> implements ITensorablePair<Float> {
    public DataPair(float[] key, float[] val) {
        super(key, val);
    }

    @Override
    public TensorPair<Float> tensorify() {
        return new TensorPair<>(Tensors.create(getKey()), Tensors.create(getVal()));
    }
}
