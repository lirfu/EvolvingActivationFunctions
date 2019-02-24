package hr.fer.zemris.evolveactivationfunction;

import hr.fer.zemris.evolveactivationfunction.nodes.*;
import hr.fer.zemris.genetics.symboregression.SymbolicTree;
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
        TreeNodeSet set = new TreeNodeSet(r) { // Modify const as a special case.
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

    public static String toLatex(SymbolicTree t) {
        String token = "<->";
        final String[] str = new String[]{token};
        t.collect(n -> {
            String s;
            switch (n.getName()) {
                case AddNode.NAME:
                    s = "(" + token + " + " + token + ")";
                    break;
                case SubNode.NAME:
                    s = "(" + token + " - " + token + ")";
                    break;
                case MulNode.NAME:
                    s = token + " \\\\cdot " + token;
                    break;
                case DivNode.NAME:
                    s = "\\\\frac{" + token + "}{" + token + "}";
                    break;
                case MinNode.NAME:
                    s = "\\\\min (" + token + "," + token + ")";
                    break;
                case MaxNode.NAME:
                    s = "\\\\max (" + token + "," + token + ")";
                    break;
                case SinNode.NAME:
                    s = "\\\\sin (" + token + ")";
                    break;
                case CosNode.NAME:
                    s = "\\\\cos (" + token + ")";
                    break;
                case TanNode.NAME:
                    s = "\\\\tan (" + token + ")";
                    break;
                case ExpNode.NAME:
                    s = "\\\\exp (" + token + ")";
                    break;
                case Pow2Node.NAME:
                    s = "(" + token + ")^2";
                    break;
                case Pow3Node.NAME:
                    s = "(" + token + ")^3";
                    break;
                case PowNode.NAME:
                    s = "(" + token + ")^{" + token + "}";
                    break;
                case LogNode.NAME:
                    s = "\\\\log (" + token + ")";
                    break;
                case InputNode.NAME:
                    s = "x";
                    break;
                case ConstNode.NAME:
                    s = String.valueOf((Double) n.getExtra());
                    break;
                case ReLUNode.NAME:
                    s = "ReLU (" + token + ")";
                    break;
                case SigmoidNode.NAME:
                    s = "\\\\sigma (" + token + ")";
                    break;
                case GaussNode.NAME:
                    s = "gauss (" + token + ")";
                    break;
                default:
                    s = "NONE";
            }
            str[0] = str[0].replaceFirst(token, s);
            return false;
        }, null);
        return str[0];
    }

    public interface Listable<T> {
        public T[] list();
    }

}
