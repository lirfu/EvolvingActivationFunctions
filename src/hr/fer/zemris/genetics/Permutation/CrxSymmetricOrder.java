package hr.fer.zemris.genetics.Permutation;

import hr.fer.zemris.genetics.Crossover;

import java.util.Random;

/**
 * Preserves the, symmetrically cut, middle of 1st parent. Replaces the rest with values from 2nd parent, checking from left to right.
 */
public class CrxSymmetricOrder extends Crossover<Permutation> {
    private Random rand;
    private float boundsPercentage;

    public CrxSymmetricOrder(Random random, float replacementPercentage) {
        rand = random;
        boundsPercentage = replacementPercentage / 2;
    }

    @Override
    public Permutation cross(Permutation parent1, Permutation parent2) {
        int size = parent1.size();
        // Symmetric cutoff like: 012|3456|789
        int x1 = rand.nextInt((int) (size * boundsPercentage - 1)) + 1;
        int x2 = size - 1 - x1;

        Permutation child = (Permutation) parent1.copy();
        int previousIndex = 0;
        for (int i = 0; i < size; i++) {

            Integer value = null;
            for (int j = previousIndex; j < size; j++) { // Iterate 2nd parent values (from the last found index+1).
                value = parent2.get(j);
                boolean different = true;

                for (int k = x1; k <= x2; k++) // Check that it's unique to selected section.
                    different &= !child.get(k).equals(value);

                if (different) { // Found a unique.
                    previousIndex = j + 1;
                    break;
                }
            }
            child.set(i, value);

            if (i == x1 - 1) i = x2; // Skip selected region.
        }

        return child;
    }
}
