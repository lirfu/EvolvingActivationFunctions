package hr.fer.zemris.genetics.symboregression.mut;

import hr.fer.zemris.genetics.Mutation;
import hr.fer.zemris.genetics.symboregression.SymbolicTree;
import hr.fer.zemris.genetics.symboregression.TreeNode;
import hr.fer.zemris.genetics.symboregression.nodes.ConstNode;

import java.util.LinkedList;
import java.util.Random;

public class MutSRRandomConstantSet extends Mutation<SymbolicTree> {
    private Random r_;
    private double min_, delta_;

    public MutSRRandomConstantSet(Random r, double min, double max) {
        r_ = r;
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

        // Set a random node.
        LinkedList<TreeNode> l = new LinkedList<>();
        genotype.collect(c, l);

        if (l.isEmpty()) return;

        l.get(r_.nextInt(l.size())).setExtra(min_ + delta_ * r_.nextDouble());
    }

    @Override
    public void parse(String line) {
        super.parse(line);
        String[] p = line.split(SPLIT_REGEX);
        if (p[0].equals(getName() + ".min")) {
            min_ = Double.parseDouble(p[1]);
        } else if (p[0].equals(getName() + ".delta")) {
            delta_ = Double.parseDouble(p[1]);
        }
    }

    @Override
    public String serialize() {
        return super.serialize()
                + serializeKeyVal(getName() + ".min", String.valueOf(min_))
                + serializeKeyVal(getName() + ".delta", String.valueOf(delta_));
    }
}
