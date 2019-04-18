package hr.fer.zemris.data.primitives;

import com.sun.istack.NotNull;
import hr.fer.zemris.utils.Pair;
import org.tensorflow.Tensor;
import org.tensorflow.Tensors;

import java.nio.FloatBuffer;

/**
 * Container for <b>Tensor</b> data pairs.
 */
public class DatumF implements IDatum<Tensor<Float>> {
    private Tensor<Float> input_;
    private Tensor<Float> label_;

    public DatumF(@NotNull DataPair pair) {
        input_ = Tensors.create(pair.getKey());
        label_ = Tensors.create(pair.getVal());
    }

    public DatumF(@NotNull BatchPair pair) {
        input_ = Tensors.create(pair.getKey());
        label_ = Tensors.create(pair.getVal());
    }

    public DatumF(@NotNull Tensor<Float> input, @NotNull Tensor<Float> label) {
        input_ = input;
        label_ = label;
    }

    public Tensor<Float> getInput() {
        return input_;
    }

    public Tensor<Float> getLabel() {
        return label_;
    }

    public String toString(@NotNull String delimiter) {
        int input_size = input_.numElements(), label_size = label_.numElements();
        FloatBuffer i_b = FloatBuffer.allocate(input_size);
        FloatBuffer l_b = FloatBuffer.allocate(label_size);
        input_.writeTo(i_b);
        label_.writeTo(l_b);

        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < label_size; i++) {
            for (int j = 0; j < input_size / label_size; j++) {
                sb.append(i_b.get(j + i * input_size / label_size)).append(delimiter);
            }
            sb.append(l_b.get(i)).append('\n');
        }
        return sb.toString();
    }
}
