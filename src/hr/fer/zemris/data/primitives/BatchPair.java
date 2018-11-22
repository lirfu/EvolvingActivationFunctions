package hr.fer.zemris.data.primitives;

import hr.fer.zemris.utils.Pair;
import org.nd4j.linalg.factory.Nd4j;
import org.tensorflow.Tensors;

import java.util.ArrayList;
import java.util.Arrays;

/**
 * Primitive for a batch of input-label pairs (matrix, vector).
 */
public class BatchPair extends Pair<float[][], float[][]> implements ITensorablePair<Float> {
    public BatchPair(float[][] key, float[][] val) {
        super(key, val);
    }

    @Override
    public TensorPair<Float> tensorify() {
        return new TensorPair<>(Tensors.create(getKey()), Tensors.create(getVal()));
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        for (float[] a : getKey()) {
            sb.append('|').append(Arrays.toString(a)).append('|').append('\n');
        }
        sb.append('<').append(Arrays.toString(getVal())).append('>');
        return sb.toString();
    }
}
