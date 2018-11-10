package hr.fer.zemris.genetics.algorithms;

import hr.fer.zemris.genetics.Algorithm;
import hr.fer.zemris.genetics.Genotype;
import hr.fer.zemris.genetics.Selector;
import hr.fer.zemris.utils.threading.Work;

import java.util.Random;

/**
 * Constructs a completely new population from the old one, replacing the old population.
 * By enabling elitism the best unit from the old population can survive and be added to the new population.
 */
public class GenerationAlgorithm extends Algorithm {
    private Random random = new Random();
    private boolean elitism = false;

    public GenerationAlgorithm(Algorithm algorithm, boolean elitism) {
        super(algorithm);
        this.elitism = elitism;
    }

    private Integer index;

    @Override
    protected void runIteration(Genotype[] population) {
        int size = population.length;

        // Create the new population.
        Genotype[] oldPopulation = new Genotype[size];
        System.arraycopy(population, 0, oldPopulation, 0, size);
        index = 0;

        // In case of capitalism, save the elite.
        if (elitism) {
            // Find the best.
            Genotype best = oldPopulation[0];
            for (Genotype g : oldPopulation)
                if (best.getFitness() > g.getFitness())
                    best = g;

            population[index++] = best;
        }

        Work work = () -> {
            if (index < size) {
                Selector.Parent[] parents = mSelector.selectParentsFrom(random, oldPopulation);

                Genotype child = getCrossover(random).cross(parents[0].getGenotype(), parents[1].getGenotype());
                getMutation(random).mutate(child);
                child.evaluate(mFitnessFunction);

                synchronized (index) {
                    if (index < size) {
                        population[index] = child;
                        index++;
                    }
                }
            }
        };

        // Fill the new population with children of old parents from old population.

        while (index < size)
            mWorkArbiter.postWork(work);
    }
}
