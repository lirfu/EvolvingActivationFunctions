package hr.fer.zemris.genetics.symboregression.mut;

import hr.fer.zemris.genetics.Mutation;
import hr.fer.zemris.genetics.Utils;
import hr.fer.zemris.genetics.symboregression.SymbolicTree;
import hr.fer.zemris.genetics.symboregression.TreeNode;
import hr.fer.zemris.genetics.symboregression.TreeNodeSet;

import java.util.Random;

/**
 * Replaces a random node with a new random node of the same order (children number).
 */
public class MutSRReplaceNode extends Mutation<SymbolicTree> {
    private final TreeNodeSet set_;
    private final Random r_;

    public MutSRReplaceNode(TreeNodeSet set, Random random) {
        set_ = set;
        r_ = random;
    }

    @Override
    public String getName() {
        return "mut.replace_node";
    }

    @Override
    public void mutate(SymbolicTree genotype) {
        TreeNode n = genotype.get(r_.nextInt(genotype.size()));
        n.swapNodeWith(set_.getRandomNode(n.getChildrenNum()));
    }
}
