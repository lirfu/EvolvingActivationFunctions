package hr.fer.zemris.genetics.symboregression.crx;

import hr.fer.zemris.genetics.Crossover;
import hr.fer.zemris.genetics.symboregression.SymbolicTree;
import hr.fer.zemris.genetics.symboregression.TreeNode;

import java.util.LinkedList;
import java.util.Random;

/**
 * Swaps randomly selected tree nodes and their subtrees.
 */
public class CrxSRSwapNodes extends Crossover<SymbolicTree> {

    public CrxSRSwapNodes() {
    }

    @Override
    public String getName() {
        return "crx.swap_nodes";
    }

    @Override
    public SymbolicTree cross(SymbolicTree parent1, SymbolicTree parent2) {
        // Construct children by copying parents.
        SymbolicTree child1;
        SymbolicTree child2;
        if (parent1.size() < parent2.size()) {
            child1 = parent1.copy();
            child2 = parent2.copy();
        } else {
            child1 = parent2.copy();
            child2 = parent1.copy();
        }

        // Select a random node from smaller tree.
        TreeNode n = child1.get(r_.nextInt(child1.size()));

        // Collect nodes of same order in larger tree (more chance of finding a match here).
        LinkedList<TreeNode> nodes = new LinkedList<>();
        child2.collect(node -> node.getChildrenNum() == n.getChildrenNum(), nodes);

        // On failure return random.
        if (nodes.isEmpty()) return r_.nextBoolean() ? child1 : child2;

        // Swap the nodes.
        n.swapInternalsWith(nodes.get(r_.nextInt(nodes.size())));

        return r_.nextBoolean() ? child1 : child2;
    }
}
