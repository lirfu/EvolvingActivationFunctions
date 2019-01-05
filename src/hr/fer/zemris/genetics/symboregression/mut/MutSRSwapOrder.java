package hr.fer.zemris.genetics.symboregression.mut;

import hr.fer.zemris.genetics.Mutation;
import hr.fer.zemris.genetics.Utils;
import hr.fer.zemris.genetics.symboregression.SymbolicTree;
import hr.fer.zemris.genetics.symboregression.TreeNode;

import java.util.Random;

/**
 * Swap children order.
 * Effective only for directed ops (e.g. a / b).
 * Ignores terminal and unary nodes.
 * Can fizzle if random selects same children indexes.
 */
public class MutSRSwapOrder extends Mutation<SymbolicTree> {
    private final Random r_;

    public MutSRSwapOrder(Random random) {
        r_ = random;
    }

    @Override
    public String getName() {
        return "mut.swap_order";
    }

    @Override
    public void mutate(SymbolicTree genotype) {
        // Get random node.
        TreeNode n = genotype.get(r_.nextInt(genotype.size()));
        int c_num = n.getChildrenNum();

        // Select children indexes to swap.
        int i1, i2;
        if (c_num == 2) { // Simple swap.
            i1 = 0;
            i2 = 1;
        } else if (c_num > 2) { // Multi children node, can fizzle if random selects the same index.
            i1 = r_.nextInt(c_num);
            i2 = r_.nextInt(c_num);
        } else { // Can't swap children of terminals or unary ops.
            return;
        }

        // Swap the selected children.
        n.getChildren()[i1].swapContentWith(n.getChildren()[i2]);
    }
}
