package hr.fer.zemris.data.modifiers;

import com.sun.istack.NotNull;
import hr.fer.zemris.data.primitives.DataPair;
import hr.fer.zemris.utils.Utilities;

import java.util.ArrayList;
import java.util.Random;

/**
 * Randomizes the order of given dataset array elements.
 * Randomizes by doing <code>n</code> swaps of randomly selected elements.
 */
public class Randomizer implements IModifier<DataPair> {
    private int n_;
    private Random r_;

    /**
     * Randomizes the order of given dataset array elements.
     * Randomizes by doing <code>n</code> swaps of randomly selected elements.
     * Creates a new random_ generator.
     *
     * @param n Number of random_ swaps performed on the dataset.
     */
    public Randomizer(int n) {
        this(n, new Random());
    }

    /**
     * Randomizes the order of given dataset array elements.
     * Randomizes by doing <code>n</code> swaps of randomly selected elements.
     * Uses the given random_ generator.
     *
     * @param n      Number of random_ swaps performed on the dataset.
     * @param random Random generator used for swapping.
     */
    public Randomizer(int n, @NotNull Random random) {
        if (n < 0) throw new IllegalArgumentException("Swaps number must not be negative.");
        n_ = n;
        r_ = random;
    }

    @Override
    public void apply(ArrayList<DataPair> data) {
//        DataPair[] arr = (DataPair[]) data.toArray();
//        Utilities.permuteArray(arr, n_, r_);
//        data.
        // FIXME If needed.
    }
}
