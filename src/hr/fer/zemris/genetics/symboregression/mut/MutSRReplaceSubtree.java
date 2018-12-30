package hr.fer.zemris.genetics.symboregression.mut;

import hr.fer.zemris.genetics.Initializer;
import hr.fer.zemris.genetics.Mutation;
import hr.fer.zemris.genetics.Utils;
import hr.fer.zemris.genetics.symboregression.SymbolicTree;
import hr.fer.zemris.genetics.symboregression.TreeNodeSet;

import java.util.Random;

/**
 * Replace a node with a random subtree.
 */
public class MutSRReplaceSubtree extends Mutation<SymbolicTree> {
    private final TreeNodeSet set_;
    private final Initializer<SymbolicTree> init_;
    private final Random r_;

    public MutSRReplaceSubtree(TreeNodeSet set, Initializer<SymbolicTree> init, Random random) {
        set_ = set;
        init_ = init;
        r_ = random;
    }

    @Override
    public void mutate(SymbolicTree genotype) {
        // Build a random subtree.
        SymbolicTree tree = new SymbolicTree(set_, null);
        init_.initialize(tree);

        int i = r_.nextInt(genotype.size());
        genotype.set(i, tree.get(0));
    }
}
