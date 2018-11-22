package hr.fer.zemris.neurology.tf;

import com.sun.istack.internal.NotNull;
import org.tensorflow.Operand;

public interface IActivationFunction<T> {
    public Operand<T> buildUpon(@NotNull Operand<T> input);
}
