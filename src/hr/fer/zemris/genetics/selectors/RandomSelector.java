package hr.fer.zemris.genetics.selectors;

import hr.fer.zemris.genetics.Genotype;
import hr.fer.zemris.genetics.Selector;

import java.util.Random;

public class RandomSelector implements Selector {

    public RandomSelector() {
    }

    @SuppressWarnings("Duplicates")
    @Override
    public Parent[] selectParentsFrom(Random rand, Genotype[] population) {
        int p1 = rand.nextInt(population.length);
        int p2 = rand.nextInt(population.length);
        int c = rand.nextInt(population.length);

        // Take first two best and replace the worst.
        return new Parent[]{new Parent(population[p1], p1), new Parent(population[p2], p2), new Parent(population[c], c)};
    }
}
