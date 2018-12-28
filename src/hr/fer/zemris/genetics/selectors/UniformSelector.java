package hr.fer.zemris.genetics.selectors;

import hr.fer.zemris.genetics.Genotype;
import hr.fer.zemris.genetics.Selector;

import java.util.Random;

/**
 * Selects units randomly from a uniform distribution.
 */
public class UniformSelector implements Selector<Genotype> {

    private Random rand;

    public UniformSelector() {
        rand = new Random();
    }

    public UniformSelector(Random r) {
        rand = r;
    }

    @SuppressWarnings("Duplicates")
    @Override
    public Parent[] selectParentsFrom(Genotype[] population) {
        int p1 = rand.nextInt(population.length);
        int p2 = rand.nextInt(population.length);
        int c1 = rand.nextInt(population.length);
        int c2 = rand.nextInt(population.length);
        return new Parent[]{new Parent<Genotype>(population[p1], p1), new Parent<Genotype>(population[p2], p2),
                new Parent<Genotype>(population[c1], c1), new Parent<Genotype>(population[c2], c2)};
    }
}
