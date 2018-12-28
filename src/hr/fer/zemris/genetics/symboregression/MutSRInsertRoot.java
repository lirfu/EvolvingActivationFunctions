package hr.fer.zemris.genetics.symboregression;

import hr.fer.zemris.genetics.Mutation;
import hr.fer.zemris.genetics.Utils;

import java.util.Random;

public class MutSRInsertRoot extends Mutation<SymbolicTree> {
    private final TreeNodeSet set_;
    private final Random r_;

    public MutSRInsertRoot(TreeNodeSet set, Random random) {
        set_ = set;
        r_ = random;
    }

    @Override
    public void mutate(SymbolicTree genotype) {
         // Create a new operator and set current root as its first child.
        TreeNode root = set_.getRandomOperator();
        root.getChildren()[0] = genotype.root_.clone();

        // Populate the rest of the children with terminals.
        for (int i = 1; i < root.getChildrenNum(); i++)
            root.getChildren()[i] = set_.getRandomTerminal();

        // Update the new tree root.
        genotype.set(0, root);
    }
}
