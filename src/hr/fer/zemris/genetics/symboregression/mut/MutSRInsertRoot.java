package hr.fer.zemris.genetics.symboregression.mut;

import hr.fer.zemris.genetics.Mutation;
import hr.fer.zemris.genetics.Utils;
import hr.fer.zemris.genetics.symboregression.SymbolicTree;
import hr.fer.zemris.genetics.symboregression.TreeNode;
import hr.fer.zemris.genetics.symboregression.TreeNodeSet;

import java.util.Random;

public class MutSRInsertRoot extends Mutation<SymbolicTree> {
    private final TreeNodeSet set_;

    public MutSRInsertRoot(TreeNodeSet set) {
        set_ = set;
    }

    @Override
    public String getName() {
        return "mut.insert_root";
    }

    @Override
    public void mutate(SymbolicTree genotype) {
         // Create a new operator and set current root as its first child.
        TreeNode root = set_.getRandomOperator();
        root.getChildren()[0] = genotype.get(0).clone();

        // Populate the rest of the children with terminals.
        for (int i = 1; i < root.getChildrenNum(); i++)
            root.getChildren()[i] = set_.getRandomTerminal();

        // Update the new tree root.
        genotype.set(0, root);
    }
}
