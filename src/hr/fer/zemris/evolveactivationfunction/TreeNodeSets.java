package hr.fer.zemris.evolveactivationfunction;

import hr.fer.zemris.evolveactivationfunction.activationfunction.DerivableNode;
import hr.fer.zemris.evolveactivationfunction.nodes.*;

import java.util.LinkedList;

public enum TreeNodeSets implements Listable<DerivableNode> {
    /* BINARY FUNCTIONS */

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
    POW {
        @Override
        public DerivableNode[] list() {
            return new DerivableNode[]{new PowNode()};
        }
    },

    /* UNARY FUNCTIONS */

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
    LOG {
        @Override
        public DerivableNode[] list() {
            return new DerivableNode[]{new LogNode()};
        }
    },

    ELU {
        @Override
        public DerivableNode[] list() {
            return new DerivableNode[]{new ELUNode()};
        }
    },
    GAUSS {
        @Override
        public DerivableNode[] list() {
            return new DerivableNode[]{new GaussNode()};
        }
    },
    LRELU {
        @Override
        public DerivableNode[] list() {
            return new DerivableNode[]{new LReLUNode()};
        }
    },
    RELU {
        @Override
        public DerivableNode[] list() {
            return new DerivableNode[]{new ReLUNode()};
        }
    },
    SELU {
        @Override
        public DerivableNode[] list() {
            return new DerivableNode[]{new SELUNode()};
        }
    },
    SIGMOID {
        @Override
        public DerivableNode[] list() {
            return new DerivableNode[]{new SigmoidNode()};
        }
    }, SOFTPLUS {
        @Override
        public DerivableNode[] list() {
            return new DerivableNode[]{new SoftPlusNode()};
        }
    },
    SOFTSIGN {
        @Override
        public DerivableNode[] list() {
            return new DerivableNode[]{new SoftSignNode()};
        }
    },
    SWISH {
        @Override
        public DerivableNode[] list() {
            return new DerivableNode[]{new SwishNode()};
        }
    },
    TANH {
        @Override
        public DerivableNode[] list() {
            return new DerivableNode[]{new TanhNode()};
        }
    },

    /* TERMINALS */

    CONSTANT {
        @Override
        public DerivableNode[] list() {
            return new DerivableNode[]{new ConstNode()};
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
            return new DerivableNode[]{ELU.list()[0], GAUSS.list()[0], LRELU.list()[0], RELU.list()[0], SELU.list()[0],
                    SIGMOID.list()[0], SOFTPLUS.list()[0], SOFTSIGN.list()[0], SWISH.list()[0], TANH.list()[0]};
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
