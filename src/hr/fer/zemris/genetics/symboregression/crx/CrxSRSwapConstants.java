package hr.fer.zemris.genetics.symboregression.crx;

import hr.fer.zemris.genetics.Crossover;
import hr.fer.zemris.genetics.symboregression.SymbolicTree;
import hr.fer.zemris.genetics.symboregression.TreeNode;
import hr.fer.zemris.genetics.symboregression.nodes.ConstNode;

import java.util.LinkedList;
import java.util.Random;

public class CrxSRSwapConstants extends Crossover<SymbolicTree> {

    public CrxSRSwapConstants() {
    }

    @Override
    public String getName() {
        return "crx.swap_constants";
    }

    @Override
    public SymbolicTree cross(SymbolicTree parent1, SymbolicTree parent2) {
        TreeNode.Condition c = node -> node.getName().equals(ConstNode.NAME);

        SymbolicTree child1 = parent1.copy();
        LinkedList<TreeNode> l1 = new LinkedList<>();
        child1.collect(c, l1);

        SymbolicTree child2 = parent2.copy();
        LinkedList<TreeNode> l2 = new LinkedList<>();
        child2.collect(c, l2);

        if (l1.isEmpty() || l2.isEmpty()) return r_.nextBoolean() ? child1 : child2;

        TreeNode n1 = l1.get(r_.nextInt(l1.size()));
        TreeNode n2 = l2.get(r_.nextInt(l2.size()));

        Object t = n1.getExtra();
        n1.setExtra(n2.getExtra());
        n2.setExtra(t);

        return r_.nextBoolean() ? child1 : child2;
    }
}
