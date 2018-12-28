package hr.fer.zemris.genetics.symboregression;

import hr.fer.zemris.genetics.Crossover;

import java.util.Random;

/**
 * Swaps randomly selected tree nodes and their subtrees.
 */
public class CrxSRSwapSubtree extends Crossover<SymbolicTree> {
    private final Random r_;

    public CrxSRSwapSubtree(Random random) {
        r_ = random;
    }

    @Override
    public SymbolicTree cross(SymbolicTree parent1, SymbolicTree parent2) {
        // Construct children by copying parents.
        SymbolicTree child1 = parent1.copy();
        SymbolicTree child2 = parent2.copy();

        // Select nodes from children and swap them.
        child1.set(r_.nextInt(child1.size()), child2.get(r_.nextInt(child2.size())));
        child2.updateSize();

        return r_.nextBoolean() ? child1 : child2;
    }
}
