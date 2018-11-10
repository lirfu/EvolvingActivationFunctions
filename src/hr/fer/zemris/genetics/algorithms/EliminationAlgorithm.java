package hr.fer.zemris.genetics.algorithms;

import hr.fer.zemris.genetics.Algorithm;
import hr.fer.zemris.genetics.Genotype;
import hr.fer.zemris.genetics.Selector;

/**
 * Each iteration performs a single tournament, replacing the worst unit with a child of better ones.
 */
public class EliminationAlgorithm extends Algorithm {

    public EliminationAlgorithm(Algorithm algorithm) {
        super(algorithm);
    }

    @Override
    protected void runIteration(Genotype[] population) {
        Selector.Parent[] parents = mSelector.selectParentsFrom(random, population);

        Genotype child = getCrossover(random).cross(parents[0].getGenotype(), parents[1].getGenotype());
        getMutation(random).mutate(child);
        child.evaluate(mFitnessFunction);

        // Find the index of the replacement.
        int indexWorst;
        if (parents[2] != null) // If selector specifies the index.
            indexWorst = parents[2].getIndex();

        else { // If not, replace the worst in the population.
            Genotype worst = population[0];
            indexWorst = 0;
            for (int i = 1; i < population.length; i++)
                if (population[i].getFitness() > worst.getFitness()) {
                    worst = population[i];
                    indexWorst = i;
                }
        }

        population[parents[0].getIndex()] = parents[0].getGenotype();
        population[parents[1].getIndex()] = parents[1].getGenotype();
        population[indexWorst] = child;
    }
}
