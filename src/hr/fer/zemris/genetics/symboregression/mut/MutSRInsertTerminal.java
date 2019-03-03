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

    public MutSRInsertTerminal(TreeNodeSet set) {
        set_ = set;
    }

    @Override
    public String getName() {
        return "mut.insert_terminal";
    }

    @Override
    public void mutate(SymbolicTree genotype) {
        genotype.set(r_.nextInt(genotype.size()), set_.getRandomTerminal());
    }
}
