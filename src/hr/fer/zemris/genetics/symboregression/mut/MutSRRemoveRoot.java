package hr.fer.zemris.genetics.symboregression.mut;

import hr.fer.zemris.genetics.Mutation;
import hr.fer.zemris.genetics.symboregression.SymbolicTree;
import hr.fer.zemris.genetics.symboregression.TreeNode;
import hr.fer.zemris.genetics.symboregression.TreeNodeSet;

import java.util.Random;

public class MutSRRemoveRoot extends Mutation<SymbolicTree> {
    private final Random r_;

    public MutSRRemoveRoot(Random random) {
        r_ = random;
    }

    @Override
    public String getName() {
        return "mut.remove_root";
    }

    @Override
    public void mutate(SymbolicTree genotype) {
        TreeNode root = genotype.get(0);

        // Don't remove if root is terminal.
        if (root.getChildrenNum() == 0) return;

        // Replace root with its random child.
        root = root.getChild(r_.nextInt(root.getChildrenNum()));
        genotype.set(0, root);
    }
}
