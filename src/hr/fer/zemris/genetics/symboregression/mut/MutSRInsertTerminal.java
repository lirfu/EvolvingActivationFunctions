package hr.fer.zemris.genetics.symboregression.mut;

import hr.fer.zemris.genetics.Mutation;
import hr.fer.zemris.genetics.Utils;
import hr.fer.zemris.genetics.symboregression.SymbolicTree;
import hr.fer.zemris.genetics.symboregression.TreeNodeSet;

import java.util.Random;

/**
 * Replaces random child with a terminal.
 */
public class MutSRInsertTerminal extends Mutation<SymbolicTree> {
    private final TreeNodeSet set_;
    private final Random r_;

    public MutSRInsertTerminal(TreeNodeSet set, Random random) {
        set_ = set;
        r_ = random;
    }

    @Override
    public void mutate(SymbolicTree genotype) {
        genotype.set(r_.nextInt(genotype.size()), set_.getRandomTerminal());
    }
}
