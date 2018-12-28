package hr.fer.zemris.genetics.algorithms;

import hr.fer.zemris.genetics.Algorithm;
import hr.fer.zemris.genetics.Genotype;
import hr.fer.zemris.genetics.Selector;

/**
 * Each iteration performs a single tournament, replacing the worst unit with a child of better ones.
 */
public class EliminationAlgorithm extends Algorithm {

    // FIXME Should be defined as: killing a given percentage of the population (double generationGap).
    public EliminationAlgorithm(Algorithm algorithm) {
        super(algorithm);
    }

    @Override
    protected void runIteration() {
        Selector.Parent[] parents = selector_.selectParentsFrom(population_);

        Genotype child = getRandomCrossover().cross(parents[0].getGenotype(), parents[1].getGenotype());
        getRandomMutation().mutate(child);
        child.evaluate(evaluator_);

        // Find the index of the replacement.
        int indexWorst;
        if (parents[2] != null) // If selector specifies the index.
            indexWorst = parents[2].getIndex();

        else { // If not, replace the worst in the population.
            Genotype worst = population_[0];
            indexWorst = 0;
            for (int i = 1; i < population_.length; i++)
                if (population_[i].getFitness() > worst.getFitness()) {
                    worst = population_[i];
                    indexWorst = i;
                }
        }

        population_[parents[0].getIndex()] = parents[0].getGenotype();
        population_[parents[1].getIndex()] = parents[1].getGenotype();
        population_[indexWorst] = child;
    }
}
