package hr.fer.zemris.evolveactivationfunction.tree;

import hr.fer.zemris.evolveactivationfunction.tree.nodes.*;
import hr.fer.zemris.genetics.symboregression.TreeNode;
import hr.fer.zemris.genetics.symboregression.TreeNodeSet;

import java.util.LinkedList;
import java.util.Random;

public class TreeNodeSetFactory {
    public static TreeNodeSet build(Random r, String... set_names) {
        LinkedList<TreeNodeSets> sets = new LinkedList<>();
        for (String s : set_names) {
            try {
                sets.add(TreeNodeSets.valueOf(s.toUpperCase()));
            } catch (IllegalArgumentException e) {
                System.err.println("Unknown set: " + s);
            }
        }
        return build(r, sets.toArray(new TreeNodeSets[]{}));
    }

    public static TreeNodeSet build(Random r, TreeNodeSets... use_sets) {
        TreeNodeSet set = new TreeNodeSet(r) { // Modify constant node as a special case.
            @Override
            public TreeNode getNode(String node_name) {
                TreeNode node = super.getNode(node_name);
                if (node == null) {
                    try {
                        Double val = Double.parseDouble(node_name);
                        node = new ConstNode();
                        node.setExtra(val);
                    } catch (NumberFormatException e) {
                    }
                }
                return node;
            }
        };
        // Input is always present.
        set.registerTerminal(new InputNode());
        // Construct from sets.
        for (TreeNodeSets s : use_sets) {
            for (TreeNode n : s.list()) {
                set.registerNode(n);
            }
        }

        return set;
    }
}
