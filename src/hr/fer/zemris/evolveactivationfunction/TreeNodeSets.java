package hr.fer.zemris.evolveactivationfunction;

import hr.fer.zemris.evolveactivationfunction.activationfunction.DerivableNode;
import hr.fer.zemris.evolveactivationfunction.nodes.*;

import java.util.LinkedList;

public enum TreeNodeSets implements TreeNodeSetFactory.Listable<DerivableNode> {
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
            for (TreeNodeSets s : TreeNodeSets.values()) {
                if (s.equals(ALL) || s.list().length > 1) continue;
                nodes.add(s.list()[0]);
            }
            return nodes.toArray(new DerivableNode[]{});
        }
    }
}
