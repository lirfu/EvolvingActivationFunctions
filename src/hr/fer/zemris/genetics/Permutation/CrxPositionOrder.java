package hr.fer.zemris.genetics.Permutation;

import hr.fer.zemris.genetics.Crossover;

import java.util.ArrayList;
import java.util.Random;

public class CrxPositionOrder extends Crossover<Permutation> {
    private Random rand;
    private float selectPercentage;

    public CrxPositionOrder(Random random, float selectPercentage) {
        rand = random;
        this.selectPercentage = selectPercentage;
    }

    @Override
    public Permutation cross(Permutation parent1, Permutation parent2) {
        int size = parent1.size();
        int selected = (int) (size * selectPercentage);

        // Select indexes to preserve.
        ArrayList<Integer> indexes = new ArrayList<>(selected);
        for (int i = 0; i < selected; i++) {
            int index = rand.nextInt(size);
            while (indexes.contains(index)) {
                index = ++index % size;
            }
            indexes.add(index);
        }

        Permutation child = (Permutation) parent1.copy();
        int previousIndex = 0;
        for (int i = 0; i < size; i++) {
            if (indexes.contains(i)) continue;

            Integer value = null;
            for (int j = previousIndex; j < size; j++) { // Iterate 2nd parent values (from the last found index+1).
                value = parent2.get(j);
                boolean different = true;

                for (int k : indexes) // Check that it's unique to selected section.
                    different &= !child.get(k).equals(value);

                if (different) { // Found a unique.
                    previousIndex = j + 1;
                    break;
                }
            }
            child.set(i, value);
        }

        return child;
    }
}
