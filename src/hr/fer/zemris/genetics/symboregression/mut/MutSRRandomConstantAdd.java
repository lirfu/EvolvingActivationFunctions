package hr.fer.zemris.genetics.symboregression.mut;

import hr.fer.zemris.genetics.Mutation;
import hr.fer.zemris.genetics.symboregression.SymbolicTree;
import hr.fer.zemris.genetics.symboregression.TreeNode;
import hr.fer.zemris.genetics.symboregression.nodes.ConstNode;
import hr.fer.zemris.utils.Utilities;

import java.util.LinkedList;

public class MutSRRandomConstantAdd extends Mutation<SymbolicTree> {
    private double max_val_;

    public MutSRRandomConstantAdd(double max_value) {
        max_val_ = max_value;
    }

    @Override
    public String getName() {
        return "mut.random_constant_add";
    }

    @Override
    public void mutate(SymbolicTree genotype) {
        TreeNode.Condition c = node -> node.getName().equals(ConstNode.NAME);

        LinkedList<TreeNode> l = new LinkedList<>();
        genotype.collect(c, l);

        if (l.isEmpty()) return;

        // Add to a random const node.
        TreeNode n = l.get(r_.nextInt(l.size()));
        n.setExtra((double) n.getExtra() + (r_.nextBoolean() ? -1 : 1) * r_.nextGaussian() * max_val_);
    }

    @Override
    public boolean parse(String line) {
        super.parse(line);
        String[] p = line.split(Utilities.KEY_VALUE_REGEX.pattern());
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
