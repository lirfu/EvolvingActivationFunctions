package hr.fer.zemris.genetics.symboregression.mut;

import hr.fer.zemris.genetics.Mutation;
import hr.fer.zemris.genetics.symboregression.SymbolicTree;
import hr.fer.zemris.genetics.symboregression.TreeNode;
import hr.fer.zemris.genetics.symboregression.nodes.ConstNode;
import scala.Int;

import java.util.LinkedList;
import java.util.Random;

public class MutSRRandomConstantSetInt extends Mutation<SymbolicTree> {
    private Random r_;
    private int min_, delta_;

    public MutSRRandomConstantSetInt(Random r, int min, int max) {
        r_ = r;
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

        // Set a random node.
        LinkedList<TreeNode> l = new LinkedList<>();
        genotype.collect(c, l);

        if (l.isEmpty()) return;

        l.get(r_.nextInt(l.size())).setExtra((double) (min_ + r_.nextInt(delta_)));
    }

    @Override
    public boolean parse(String line) {
        super.parse(line);
        String[] p = line.split(SPLIT_REGEX);
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
