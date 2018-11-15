package hr.fer.zemris.data.modifiers;

import com.sun.istack.internal.NotNull;
import hr.fer.zemris.data.primitives.DataPair;
import hr.fer.zemris.utils.Pair;

import java.util.Random;

/**
 * Randomizes the order of given dataset array elements.
 * Randomizes by doing <code>n</code> swaps of randomly selected elements.
 */
public class Randomizer implements IModifier {
    private int n_;
    private Random r_;

    /**
     * Randomizes the order of given dataset array elements.
     * Randomizes by doing <code>n</code> swaps of randomly selected elements.
     * Creates a new random generator.
     *
     * @param n Number of random swaps performed on the dataset.
     */
    public Randomizer(int n) {
        this(n, new Random());
    }

    /**
     * Randomizes the order of given dataset array elements.
     * Randomizes by doing <code>n</code> swaps of randomly selected elements.
     * Uses the given random generator.
     *
     * @param n      Number of random swaps performed on the dataset.
     * @param random Random generator used for swapping.
     */
    public Randomizer(int n, @NotNull Random random) {
        if (n < 0) throw new IllegalArgumentException("Swaps number must not be negative.");
        n_ = n;
        r_ = random;
    }

    @Override
    public void apply(DataPair[] data) {
        int len = data.length;
        for (int i = 0; i < n_; i++) {
            int a = r_.nextInt(len);
            int b = r_.nextInt(len);
            DataPair t = data[a];
            data[a] = data[b];
            data[b] = t;
        }
    }
}
