package hr.fer.zemris.genetics.symboregression.mut;

import hr.fer.zemris.genetics.Mutation;
import hr.fer.zemris.genetics.symboregression.SymbolicTree;
import hr.fer.zemris.genetics.symboregression.TreeNode;
import hr.fer.zemris.genetics.symboregression.nodes.ConstNode;
import hr.fer.zemris.utils.Utilities;

import java.util.LinkedList;

public class MutSRRandomConstantSet extends Mutation<SymbolicTree> {
    private double min_, delta_;

    public MutSRRandomConstantSet(double min, double max) {
        min_ = min;
        delta_ = max - min;
    }

    @Override
    public String getName() {
        return "mut.random_constant_set";
    }

    @Override
    public void mutate(SymbolicTree genotype) {
        TreeNode.Condition c = node -> node.getName().equals(ConstNode.NAME);

        LinkedList<TreeNode> l = new LinkedList<>();
        genotype.collect(c, l);

        if (l.isEmpty()) return;

        // Set a random const node.
        l.get(r_.nextInt(l.size())).setExtra(min_ + delta_ * r_.nextDouble());
    }

    @Override
    public boolean parse(String line) {
        super.parse(line);
        String[] p = line.split(Utilities.KEY_VALUE_REGEX.pattern());
        if (p[0].equals(getName() + ".min")) {
            min_ = Double.parseDouble(p[1]);
            return true;
        } else if (p[0].equals(getName() + ".delta")) {
            delta_ = Double.parseDouble(p[1]);
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
