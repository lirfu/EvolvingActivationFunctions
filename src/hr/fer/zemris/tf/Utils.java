package hr.fer.zemris.tf;

import com.sun.istack.internal.NotNull;
import org.tensorflow.Tensor;

import java.nio.*;
import java.util.Arrays;

public class Utils {
    public static String toString(@NotNull Tensor<?> tensor, Class type) {
        int size = tensor.numElements();
        if (type == Integer.class) {
            IntBuffer buff = IntBuffer.allocate(size);
            ((Tensor<Integer>) tensor).writeTo(buff);
            return Arrays.toString(buff.array());
        } else if (type == Long.class) {
            LongBuffer buff = LongBuffer.allocate(size);
            ((Tensor<Long>) tensor).writeTo(buff);
            return Arrays.toString(buff.array());
        } else if (type == Float.class) {
            FloatBuffer buff = FloatBuffer.allocate(size);
            ((Tensor<Float>) tensor).writeTo(buff);
            return Arrays.toString(buff.array());
        } else if (type == Double.class) {
            DoubleBuffer buff = DoubleBuffer.allocate(size);
            ((Tensor<Double>) tensor).writeTo(buff);
            return Arrays.toString(buff.array());
        } else {
            throw new IllegalArgumentException("Illegal type: " + type);
        }
    }
}
