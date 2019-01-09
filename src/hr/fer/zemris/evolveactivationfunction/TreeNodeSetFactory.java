package hr.fer.zemris.evolveactivationfunction;

import hr.fer.zemris.evolveactivationfunction.activationfunction.DerivableNode;
import hr.fer.zemris.evolveactivationfunction.nodes.*;
import hr.fer.zemris.genetics.symboregression.TreeNode;
import hr.fer.zemris.genetics.symboregression.TreeNodeSet;

import java.util.LinkedList;
import java.util.Random;

public class TreeNodeSetFactory {
    public TreeNodeSet build(Random r, String... set_names) {
        LinkedList<Set> sets = new LinkedList<>();
        for (String s : set_names) {
            try {
                sets.add(Set.valueOf(s));
            } catch (IllegalArgumentException e) {
                System.err.println("Unknown set: " + s);
            }
        }
        return build(r, sets.toArray(new Set[]{}));
    }

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
        // Input is always present.
        set.registerTerminal(new InputNode());
        // Construct from sets.
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

    public enum Set implements Listable<DerivableNode> {
        /**
         * Constructs a list from all existing sets, with duplicates filtered out.
         */
        ALL {
            @Override
            public DerivableNode[] list() {
                LinkedList<DerivableNode> nodes = new LinkedList<>();
                for (Set s : Set.values()) {
                    if (s.equals(ALL)) continue;
                    for (DerivableNode n : s.list())
                        if (!nodes.contains(n))
                            nodes.add(n);
                }
                return nodes.toArray(new DerivableNode[]{});
            }

            @Override
            public String toString() {
                return "ALL";
            }
        },
        ARITHMETICS {
            @Override
            public DerivableNode[] list() {
                return new DerivableNode[]{new AddNode(), new SubNode(), new MulNode(), new DivNode()};
            }

            @Override
            public String toString() {
                return "ARITHMETICS";
            }
        },
        MATH {
            @Override
            public DerivableNode[] list() {
                return new DerivableNode[]{new MaxNode(), new MinNode()};
            }

            @Override
            public String toString() {
                return "MATH";
            }
        },
        TRIGONOMETRY {
            @Override
            public DerivableNode[] list() {
                return new DerivableNode[]{new SinNode(), new CosNode(), new TanNode()};
            }

            @Override
            public String toString() {
                return "TRIGONOMETRY";
            }
        },
        EXPONENTIALS {
            @Override
            public DerivableNode[] list() {
                return new DerivableNode[]{
                        new ExpNode(), new Pow2Node(), new Pow3Node(), new PowNode(), new LogNode()
                };
            }

            @Override
            public String toString() {
                return "EXPONENTIALS";
            }
        },
        CONSTANT {
            @Override
            public DerivableNode[] list() {
                return new DerivableNode[]{new ConstNode()};
            }

            @Override
            public String toString() {
                return "CONSTANT";
            }
        },
        ACTIVATIONS {
            @Override
            public DerivableNode[] list() {
                return new DerivableNode[]{new ReLUNode(), new SigmoidNode(), new GaussNode()};
            }

            @Override
            public String toString() {
                return "ACTIVATIONS";
            }
        }
    }
}
