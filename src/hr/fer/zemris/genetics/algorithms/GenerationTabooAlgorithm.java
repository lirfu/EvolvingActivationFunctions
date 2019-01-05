package hr.fer.zemris.genetics.algorithms;

import hr.fer.zemris.genetics.Algorithm;
import hr.fer.zemris.genetics.Genotype;
import hr.fer.zemris.genetics.Selector;
import hr.fer.zemris.genetics.Utils;
import hr.fer.zemris.utils.threading.Work;

import java.util.LinkedList;

/**
 * Constructs a completely new population from the old one, replacing the old population.
 * By enabling elitism the best unit from the old population can survive and be added to the new population.
 */
public class GenerationTabooAlgorithm extends Algorithm {
    private final boolean elitism_;
    private final int taboo_size_;
    private final int taboo_attempts_;
    private final LinkedList<String> tabu_list_;

    public GenerationTabooAlgorithm(Algorithm algorithm, boolean elitism, int taboo_size, int taboo_attempts) {
        super(algorithm);
        elitism_ = elitism;
        taboo_size_ = taboo_size;
        taboo_attempts_ = taboo_attempts;
        tabu_list_ = new LinkedList<>();
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
            // Try fixing a taboo child by mutating.
            synchronized (tabu_list_) {
                for (int i = 0; i < taboo_attempts_; i++) {
                    if (!tabu_list_.contains(child.serialize())) {
                        // Update taboo list.
                        tabu_list_.add(child.serialize());
                        if (tabu_list_.size() > taboo_size_) {
                            tabu_list_.removeFirst();
                        }
                        break;
                    }

                    getRandomMutation().mutate(child);
                }
            }
            // Evaluate.
            child.evaluate(evaluator_);

            synchronized (index) {
                // Store the child to current index.
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

    public static class Builder extends Algorithm.Builder {
        private boolean elitism_;
        private int taboo_size_;
        private int taboo_attempts_;

        @Override
        public GenerationTabooAlgorithm build() {
            return new GenerationTabooAlgorithm(super.build(), elitism_, taboo_size_, taboo_attempts_);
        }

        public Builder setElitism(boolean elitism) {
            elitism_ = elitism;
            return this;
        }

        public Builder setTabooSize(int taboo_size) {
            taboo_size_ = taboo_size;
            return this;
        }

        public Builder setTabooAttempts(int taboo_attempts) {
            taboo_attempts_ = taboo_attempts;
            return this;
        }
    }
}
