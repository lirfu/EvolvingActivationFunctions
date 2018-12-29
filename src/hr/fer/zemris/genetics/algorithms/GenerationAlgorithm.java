package hr.fer.zemris.genetics.algorithms;

import hr.fer.zemris.genetics.Algorithm;
import hr.fer.zemris.genetics.Genotype;
import hr.fer.zemris.genetics.Selector;
import hr.fer.zemris.genetics.Utils;
import hr.fer.zemris.utils.threading.Work;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.Random;

/**
 * Constructs a completely new population from the old one, replacing the old population.
 * By enabling elitism the best unit from the old population can survive and be added to the new population.
 */
public class GenerationAlgorithm extends Algorithm {
    private final boolean elitism_;

    public GenerationAlgorithm(Algorithm algorithm, boolean elitism) {
        super(algorithm);
        elitism_ = elitism;
    }

    /**
     * Internal index for replacing the old population.
     */
    private final int[] index = new int[1];

    @Override
    protected void runIteration() {
        // Store original population.
        Genotype[] original_population = population_;
        index[0] = 0;

        // Create the new population.
        int size = population_.length;
        population_ = new Genotype[size];

        // Save the queen.
        if (elitism_) {
            Genotype best = findBest(original_population);
            population_[index[0]++] = best;
        }

        // Parallelised work.
        Work work = () -> {
            // Select parents from original population.
            Selector.Parent[] parents = selector_.selectParentsFrom(original_population);
            // Process child.
            Genotype child = getRandomCrossover().cross(parents[0].getGenotype(), parents[1].getGenotype());
            if (Utils.willOccur(mut_prob_, random_)) {
                getRandomMutation().mutate(child);
            }
            // Evaluate.
            child.evaluate(evaluator_);
            // Store the child to current index.
            synchronized (index) {
                population_[index[0]] = child;
                index[0]++;
            }
        };

        // Replace the whole population with children.
        for (int i = elitism_ ? 1 : 0; i < size; i++) {
            work_arbiter_.postWork(work);
        }

        // Wait for all work to be done.
        work_arbiter_.waitOn(() -> index[0] == size);
    }
}
