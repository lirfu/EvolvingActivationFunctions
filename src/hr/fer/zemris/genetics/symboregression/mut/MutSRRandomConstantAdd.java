package hr.fer.zemris.genetics.symboregression.mut;

import hr.fer.zemris.genetics.Mutation;
import hr.fer.zemris.genetics.symboregression.SymbolicTree;
import hr.fer.zemris.genetics.symboregression.TreeNode;
import hr.fer.zemris.genetics.symboregression.nodes.ConstNode;
import hr.fer.zemris.utils.Utilities;

import java.util.Arrays;
import java.util.LinkedList;
import java.util.Random;

public class MutSRRandomConstantAdd extends Mutation<SymbolicTree> {
    private Random r_;
    private double max_val_;

    public MutSRRandomConstantAdd(Random r, double max_value) {
        r_ = r;
        max_val_ = max_value;
    }

    @Override
    public String getName() {
        return "mut.random_constant_add";
    }

    @Override
    public void mutate(SymbolicTree genotype) {
        TreeNode.Condition c = node -> node.getName().equals(ConstNode.NAME);

        // Add to a random node.
        LinkedList<TreeNode> l = new LinkedList<>();
        genotype.collect(c, l);

        if (l.isEmpty()) return;

        TreeNode n = l.get(r_.nextInt(l.size()));
        n.setExtra((double) n.getExtra() + (r_.nextBoolean() ? -1 : 1) * r_.nextDouble() * max_val_);
    }

    @Override
    public boolean parse(String line) {
        super.parse(line);
        String[] p = line.split(Utilities.PARSER_REGEX);
        if (p[0].equals(getName() + ".max")) {
            max_val_ = Double.parseDouble(p[1]);
            return true;
        }
        return false;
    }

    @Override
    public String serialize() {
        return super.serialize() + serializeKeyVal(getName() + ".max", String.valueOf(max_val_));
    }
}
