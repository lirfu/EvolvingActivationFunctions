package hr.fer.zemris.evolveactivationfunction;

import hr.fer.zemris.evolveactivationfunction.activationfunction.DerivableNode;
import hr.fer.zemris.evolveactivationfunction.nodes.*;
import hr.fer.zemris.genetics.symboregression.TreeNode;
import hr.fer.zemris.genetics.symboregression.TreeNodeSet;

import java.util.LinkedList;
import java.util.Random;

public class TreeNodeSetFactory {
    public static TreeNodeSet build(Random r, String... set_names) {
        LinkedList<Set> sets = new LinkedList<>();
        for (String s : set_names) {
            try {
                sets.add(Set.valueOf(s.toUpperCase()));
            } catch (IllegalArgumentException e) {
                System.err.println("Unknown set: " + s);
            }
        }
        return build(r, sets.toArray(new Set[]{}));
    }

    public static TreeNodeSet build(Random r, Set... use_sets) {
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
        /* PURE OP SETS */

        ADD {
            @Override
            public DerivableNode[] list() {
                return new DerivableNode[]{new AddNode()};
            }
        },
        SUB {
            @Override
            public DerivableNode[] list() {
                return new DerivableNode[]{new SubNode()};
            }
        },
        MUL {
            @Override
            public DerivableNode[] list() {
                return new DerivableNode[]{new MulNode()};
            }
        },
        DIV {
            @Override
            public DerivableNode[] list() {
                return new DerivableNode[]{new DivNode()};
            }
        },
        MIN {
            @Override
            public DerivableNode[] list() {
                return new DerivableNode[]{new MinNode()};
            }
        },
        MAX {
            @Override
            public DerivableNode[] list() {
                return new DerivableNode[]{new MaxNode()};
            }
        },
        SIN {
            @Override
            public DerivableNode[] list() {
                return new DerivableNode[]{new SinNode()};
            }
        },
        COS {
            @Override
            public DerivableNode[] list() {
                return new DerivableNode[]{new CosNode()};
            }
        },
        TAN {
            @Override
            public DerivableNode[] list() {
                return new DerivableNode[]{new TanNode()};
            }
        },
        EXP {
            @Override
            public DerivableNode[] list() {
                return new DerivableNode[]{new ExpNode()};
            }
        },
        POW2 {
            @Override
            public DerivableNode[] list() {
                return new DerivableNode[]{new Pow2Node()};
            }
        },
        POW3 {
            @Override
            public DerivableNode[] list() {
                return new DerivableNode[]{new Pow3Node()};
            }
        },
        POW {
            @Override
            public DerivableNode[] list() {
                return new DerivableNode[]{new PowNode()};
            }
        },
        LOG {
            @Override
            public DerivableNode[] list() {
                return new DerivableNode[]{new LogNode()};
            }
        },
        CONSTANT {
            @Override
            public DerivableNode[] list() {
                return new DerivableNode[]{new ConstNode()};
            }
        },
        RELU {
            @Override
            public DerivableNode[] list() {
                return new DerivableNode[]{new ReLUNode()};
            }
        },
        SIGMOID {
            @Override
            public DerivableNode[] list() {
                return new DerivableNode[]{new SigmoidNode()};
            }
        },
        GAUSS {
            @Override
            public DerivableNode[] list() {
                return new DerivableNode[]{new GaussNode()};
            }
        },

        /* PRE-MADE SETS */

        ARITHMETICS {
            @Override
            public DerivableNode[] list() {
                return new DerivableNode[]{ADD.list()[0], SUB.list()[0], MUL.list()[0], DIV.list()[0]};
            }
        },
        MATHS {
            @Override
            public DerivableNode[] list() {
                return new DerivableNode[]{MAX.list()[0], MIN.list()[0]};
            }
        },
        TRIGONOMETRY {
            @Override
            public DerivableNode[] list() {
                return new DerivableNode[]{SIN.list()[0], COS.list()[0], TAN.list()[0]};
            }
        },
        EXPONENTIALS {
            @Override
            public DerivableNode[] list() {
                return new DerivableNode[]{EXP.list()[0], POW2.list()[0], POW3.list()[0], POW.list()[0], LOG.list()[0]
                };
            }
        },
        ACTIVATIONS {
            @Override
            public DerivableNode[] list() {
                return new DerivableNode[]{RELU.list()[0], SIGMOID.list()[0], GAUSS.list()[0]};
            }
        },

        /* THE ALL SET */

        /**
         * Constructs a list from entries containing only one value (pure sets).
         */
        ALL {
            @Override
            public DerivableNode[] list() {
                LinkedList<DerivableNode> nodes = new LinkedList<>();
                for (Set s : Set.values()) {
                    if (s.list().length > 1) continue;
                    nodes.add(s.list()[0]);
                }
                return nodes.toArray(new DerivableNode[]{});
            }
        }
    }
}
