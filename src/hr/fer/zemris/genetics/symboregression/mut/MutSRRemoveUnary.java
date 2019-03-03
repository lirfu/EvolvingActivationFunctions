package hr.fer.zemris.genetics.symboregression.mut;

import hr.fer.zemris.genetics.Mutation;
import hr.fer.zemris.genetics.symboregression.SymbolicTree;
import hr.fer.zemris.genetics.symboregression.TreeNode;

import java.util.LinkedList;
import java.util.Random;

public class MutSRRemoveUnary extends Mutation<SymbolicTree> {

    public MutSRRemoveUnary() {
    }

    @Override
    public String getName() {
        return "mut.remove_unary";
    }

    @Override
    public void mutate(SymbolicTree genotype) {
        TreeNode.Condition c = node -> node.getChildrenNum() == 1;

        LinkedList<TreeNode> l = new LinkedList<>();
        genotype.collect(c, l);

        if (l.isEmpty()) return;

        // Swap unary with its child, making it disposable.
        TreeNode n = l.get(r_.nextInt(l.size()));
        n.swapAllWith(n.getChild(0));
        genotype.updateSize();
    }
}
