package hr.fer.zemris.genetics.symboregression.crx;

import hr.fer.zemris.genetics.Crossover;
import hr.fer.zemris.genetics.symboregression.SymbolicTree;

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
    public String getName() {
        return "crx.swap_subtree";
    }

    @Override
    public SymbolicTree cross(SymbolicTree parent1, SymbolicTree parent2) {
        // Construct children by copying parents.
        SymbolicTree child1 = parent1.copy();
        SymbolicTree child2 = parent2.copy();

        // Select nodes from children and swap them.
        child1.get(r_.nextInt(child1.size())).swapContentWith(child2.get(r_.nextInt(child2.size())));
        child1.updateSize();
        child2.updateSize();

        return r_.nextBoolean() ? child1 : child2;
    }
}
