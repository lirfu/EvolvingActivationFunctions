package hr.fer.zemris.genetics.symboregression.mut;

import hr.fer.zemris.genetics.Mutation;
import hr.fer.zemris.genetics.symboregression.SymbolicTree;
import hr.fer.zemris.genetics.symboregression.TreeNode;
import hr.fer.zemris.genetics.symboregression.nodes.ConstNode;
import hr.fer.zemris.utils.Utilities;

import java.util.LinkedList;

public class MutSRRandomConstantSetInt extends Mutation<SymbolicTree> {
    private int min_, delta_;

    public MutSRRandomConstantSetInt(int min, int max) {
        min_ = min;
        delta_ = max - min + 1;
    }

    @Override
    public String getName() {
        return "mut.random_constant_set_int";
    }

    @Override
    public void mutate(SymbolicTree genotype) {
        TreeNode.Condition c = node -> node.getName().equals(ConstNode.NAME);

        LinkedList<TreeNode> l = new LinkedList<>();
        genotype.collect(c, l);

        if (l.isEmpty()) return;

        // Set a random node.
        l.get(r_.nextInt(l.size())).setExtra((double) (min_ + r_.nextInt(delta_)));
    }

    @Override
    public boolean parse(String line) {
        super.parse(line);
        String[] p = line.split(Utilities.KEY_VALUE_SIMPLE_REGEX);
        if (p[0].equals(getName() + ".min")) {
            min_ = Integer.parseInt(p[1]);
            return true;
        } else if (p[0].equals(getName() + ".delta")) {
            delta_ = Integer.parseInt(p[1]);
            return true;
        }
        return false;
    }

    @Override
    public String serialize() {
        return super.serialize()
                + serializeKeyVal(getName() + ".min", String.valueOf(min_))
                + serializeKeyVal(getName() + ".delta", String.valueOf(delta_));
    }
}
