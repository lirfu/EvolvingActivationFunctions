package hr.fer.zemris.evolveactivationfunction;

import hr.fer.zemris.evolveactivationfunction.nodes.*;
import hr.fer.zemris.genetics.symboregression.TreeNode;
import hr.fer.zemris.genetics.symboregression.TreeNodeSet;

import java.util.Random;

public class TreeNodeSetFactory {
    public TreeNodeSet build(Random r, Set... use_sets) {
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

        set.registerTerminal(new InputNode());

        for (Set s : use_sets) {
            for (TreeNode n : s.list()) {
                set.registerNode(n);
            }
        }

        return set;
    }

    public interface Listable<T> {
        public T[] list();
    }

    public enum Set implements Listable<TreeNode> {
        ARITHMETICS {
            @Override
            public TreeNode[] list() {
                return new TreeNode[]{new AddNode(), new SubNode(), new MulNode(), new DivNode()};
            }
        },
        TRIGONOMETRY {
            @Override
            public TreeNode[] list() {
                return new TreeNode[]{new SinNode(), new CosNode(), new TanNode()};
            }
        },
        EXPONENTIALS {
            @Override
            public TreeNode[] list() {
                return new TreeNode[]{
                        new ExpNode(), new Pow2Node(), new Pow3Node(), new PowNode(), new LogNode()
                };
            }
        },
        CONSTANT {
            @Override
            public TreeNode[] list() {
                return new TreeNode[]{new ConstNode()};
            }
        },
        ACTIVATIONS {
            @Override
            public TreeNode[] list() {
                return new TreeNode[]{new ReLUNode(), new SigmoidNode(), new GaussNode()};
            }
        }
    }
}
