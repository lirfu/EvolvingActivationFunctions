package hr.fer.zemris.evolveactivationfunction;

import hr.fer.zemris.evolveactivationfunction.tree.nodes.DerivableNode;
import hr.fer.zemris.evolveactivationfunction.tree.nodes.*;
import hr.fer.zemris.evolveactivationfunction.tree.DerivableSymbolicTree;
import hr.fer.zemris.evolveactivationfunction.tree.TreeNodeSetFactory;
import hr.fer.zemris.evolveactivationfunction.tree.TreeNodeSets;
import hr.fer.zemris.genetics.symboregression.SymbolicTree;
import hr.fer.zemris.genetics.symboregression.TreeNodeSet;
import org.junit.BeforeClass;
import org.junit.Test;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.activations.impl.*;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Random;

import static org.junit.Assert.assertTrue;


public class NNSRTest {

    @BeforeClass
    public static void Before() {
        // Must set precision to double.
        Nd4j.setDataType(DataBuffer.Type.DOUBLE);
    }

    @Test
    public void testParsing() {
        TreeNodeSetFactory factory = new TreeNodeSetFactory();
        TreeNodeSet set = factory.build(new Random(42), TreeNodeSets.ALL);

        String string = "+[x,*[x,sin[*[-273.15,x]]]]";
        SymbolicTree tree = SymbolicTree.parse(string, set);

        assertTrue("Serialized tree must equal the inputted parse string.", tree.toString().equals(string));
    }

    private interface IFunc {
        public double f(double x);
    }

    private final double REQUIRED_PRECISION = 1e-9;

    private void testNodeEval(DerivableNode n, IFunc f, double start, double end) {
        assertTrue("Test boundaries don't satisfy: " + start + "<" + end, start < end);

        for (double x = start; x <= end; x += 0.01) {
            INDArray input = Nd4j.create(new double[]{x, 0.1 * x});
            INDArray t = Nd4j.create(new double[]{f.f(x), f.f(0.1 * x)});
            INDArray p = n.execute(input);

            assertTrue("[" + n.getName() + "] evaluation must not change shape: " + p.shape()[1] + " != 2",
                    p.shape()[1] == 2);

            assertTrue("[" + n.getName() + "] evaluation must not change the input matrix: "
                            + Nd4j.create(new double[]{x, 0.1 * x}) + "!=" + input.toString(),
                    Math.abs(x - input.getDouble(0)) == 0
                            && Math.abs(0.1 * x - input.getDouble(1)) <= REQUIRED_PRECISION);

            assertTrue("[" + n.getName() + "] should evaluate correctly: " + p + "!=" + t,
                    Math.abs(t.getDouble(0) - p.getDouble(0)) <= REQUIRED_PRECISION
                            && Math.abs(t.getDouble(1) - p.getDouble(1)) <= REQUIRED_PRECISION);
        }

        System.out.println(n.toString()+" evaluation is OK!");
    }

    private void testNodeDeriv(DerivableNode n, IFunc f, double start, double end) {
        assertTrue("Test boundaries don't satisfy: " + start + "<" + end, start < end);

        for (double x = start; x <= end; x += 0.01) {
            INDArray input = Nd4j.create(new double[]{x, 0.1 * x});
            INDArray t = Nd4j.create(new double[]{f.f(x), f.f(0.1 * x)});
            INDArray p = n.derivate(input);

            assertTrue("[" + n.getName() + "] derivative must not change shape: " + p.shape()[1] + " != 2",
                    p.shape()[1] == 2);

            assertTrue("[" + n.getName() + "] derivative must not change the input matrix: "
                            + Nd4j.create(new double[]{x, 0.1 * x}) + "!=" + input.toString(),
                    Math.abs(x - input.getDouble(0)) <= REQUIRED_PRECISION
                            && Math.abs(0.1 * x - input.getDouble(1)) <= REQUIRED_PRECISION);

            assertTrue("[" + n.getName() + "] should derivate correctly for " + x + ": " + p + "!=" + t,
                    Math.abs(t.getDouble(0) - p.getDouble(0)) <= REQUIRED_PRECISION
                            && Math.abs(t.getDouble(1) - p.getDouble(1)) <= REQUIRED_PRECISION);
        }

        System.out.println(n.toString()+" derivative is OK!");
    }

    private void testDL4Jnode(DerivableNode n, IActivation a, double start, double end) {
        assertTrue("Test boundaries don't satisfy: " + start + "<" + end, start < end);

        for (double x = start; x <= end; x += 0.01) {
            INDArray input = Nd4j.create(new double[]{x, 0.1 * x});
            String s = input.toString();

            INDArray t = a.getActivation(input.dup(), false);
            INDArray p = n.execute(input);

            assertTrue("[" + n.getName() + "] evaluation must not change shape: " + p.shape()[1] + " != 2",
                    p.shape()[1] == 2);

            assertTrue("[" + n.getName() + "] evaluation must not change the input matrix: "
                            + s + "!=" + input.toString(), s.equals(input.toString()));

            assertTrue("[" + n.getName() + "] should evaluate correctly: " + p + "!=" + t,
                    Math.abs(t.getDouble(0) - p.getDouble(0)) <= REQUIRED_PRECISION
                            && Math.abs(t.getDouble(1) - p.getDouble(1)) <= REQUIRED_PRECISION);

            t = a.backprop(input.dup(), Nd4j.create(new float[]{1f, 1f})).getFirst();
            p = n.derivate(input);

            assertTrue("[" + n.getName() + "] derivative must not change shape: " + p.shape()[1] + " != 2",
                    p.shape()[1] == 2);

            assertTrue("[" + n.getName() + "] derivative must not change the input matrix: "
                            + s + "!=" + input.toString(), s.equals(input.toString()));

            assertTrue("[" + n.getName() + "] should derivate correctly for " + x + ": " + p + "!=" + t,
                    Math.abs(t.getDouble(0) - p.getDouble(0)) <= REQUIRED_PRECISION
                            && Math.abs(t.getDouble(1) - p.getDouble(1)) <= REQUIRED_PRECISION);
        }

        System.out.println(n.toString()+" is OK!");
    }

    private DerivableNode initNode(DerivableNode n) {
        for (int i = 0; i < n.getChildrenNum(); i++) {
            n.getChildren()[i] = new InputNode();
        }
        return n;
    }

    private DerivableNode initOp(DerivableNode n, DerivableNode n2) {
        for (int i = 0; i < n.getChildrenNum(); i++) {
            n.getChildren()[i] = new InputNode();
        }
        n.getChildren()[1] = n2;
        return n;
    }

    @Test
    public void testNodes() {
        /* EVALUATIONS */

        // Input node.
        testNodeEval(new InputNode(), x -> x, -1, 1);
        ConstNode constNode = new ConstNode();
        constNode.setExtra(-273.15);
        testNodeEval(constNode, x -> -273.15, -0.1, 0.1);
        // Arithmetic.
        testNodeEval(initNode(new AddNode()), x -> x + x, -1, 1);
        testNodeEval(initNode(new SubNode()), x -> x - x, -1, 1);
        testNodeEval(initNode(new MulNode()), x -> x * x, -1, 1);
        testNodeEval(initOp(new DivNode(), new ConstNode(.1)), x -> x / (.1 + DivNode.STABILITY_CONST), 1, 3);
        testNodeEval(initOp(new MaxNode(), new ConstNode(.1)), x -> Math.max(x, .1), -1, 1);
        testNodeEval(initOp(new MinNode(), new ConstNode(.1)), x -> Math.min(x, .1), -1, 1);
        // Trigonometry.
        testNodeEval(initNode(new SinNode()), Math::sin, -1, 1);
        testNodeEval(initNode(new CosNode()), Math::cos, -1, 1);
        testNodeEval(initNode(new TanNode()), Math::tan, -1, 1);
        testNodeEval(initNode(new TrCosNode()), x -> Math.cos(Math.min(Math.PI / 2, Math.max(-Math.PI / 2, x))), -1, 1);
        // Exponentials.
        testNodeEval(initNode(new ExpNode()), Math::exp, -1, 1);
        testNodeEval(initNode(new Pow2Node()), x -> Math.pow(x, 2), -1, 1);
        testNodeEval(initNode(new Pow3Node()), x -> Math.pow(x, 3), -1, 1);
        testNodeEval(initNode(new PowNode()), x -> Math.pow(x, x), 0, 2);
        testNodeEval(initNode(new LogNode()), Math::log, 1e-3, 2);
        // Other.
        testNodeEval(initNode(new GaussNode()), x -> Math.exp(-x * x), -3, 3);
        testNodeEval(initNode(new AbsNode()), Math::abs, -3, 3);

        /* DERIVATIONS */

        // Input node.
        testNodeDeriv(new InputNode(), x -> 1., -1, 1);
        testNodeDeriv(constNode, x -> 0, -0.1, 0.1);
        // Arithmetic.
        testNodeDeriv(initNode(new AddNode()), x -> 2., -1, 1);
        testNodeDeriv(initNode(new SubNode()), x -> 0., -1, 1);
        testNodeDeriv(initNode(new MulNode()), x -> 2. * x, -1, 1);
        testNodeDeriv(initOp(new DivNode(), new ConstNode(.1)), x -> 10., 1, 3);
        testNodeDeriv(initOp(new MaxNode(), new ConstNode(.1)), x -> x > .1 ? 1 : 0, -1, 1);
        testNodeDeriv(initOp(new MinNode(), new ConstNode(.1)), x -> x < .1 ? 1 : 0, -1, 1);
        // Trigonometry.
        testNodeDeriv(initNode(new SinNode()), Math::cos, -1, 1);
        testNodeDeriv(initNode(new CosNode()), x -> -1. * Math.sin(x), -1, 1);
        testNodeDeriv(initNode(new TanNode()), x -> 1. / Math.cos(x) / Math.cos(x), -1, 1);
        // Exponentials.
        testNodeDeriv(initNode(new ExpNode()), Math::exp, -1, 1);
        testNodeDeriv(initNode(new Pow2Node()), x -> 2 * x, -1, 1);
        testNodeDeriv(initNode(new Pow3Node()), x -> 3 * Math.pow(x, 2), -1, 1);
        testNodeDeriv(initNode(new PowNode()), x -> Math.pow(x, x) * (1 + Math.log(x)), 1, 2);
        testNodeDeriv(initNode(new LogNode()), x -> 1. / x, 1e-3, 2);
        // Other.
        testNodeDeriv(initNode(new GaussNode()), x -> Math.exp(-x * x) * (-2 * x), -3, 3);
        testNodeDeriv(initNode(new AbsNode()), x -> (Math.abs(x) / (x + AbsNode.STABILITY_CONST)), -3, 3);

        // DL4J.
        testDL4Jnode(initNode(new ReLUNode()),new ActivationReLU(),-1,1);
        testDL4Jnode(initNode(new LReLUNode()),new ActivationLReLU(),-1,1);
        testDL4Jnode(initNode(new ThReLUNode()),new ActivationThresholdedReLU(),-1,1);
        testDL4Jnode(initNode(new ELUNode()),new ActivationELU(),-1,1);
        testDL4Jnode(initNode(new SELUNode()),new ActivationSELU(),-1,1);
        testDL4Jnode(initNode(new SigmoidNode()),new ActivationSigmoid(),-1,1);
        testDL4Jnode(initNode(new HardSigmoidNode()),new ActivationHardSigmoid(),-1,1);
        testDL4Jnode(initNode(new SoftmaxNode()),new ActivationSoftmax(),-1,1);
        testDL4Jnode(initNode(new SoftplusNode()),new ActivationSoftPlus(),-1,1);
        testDL4Jnode(initNode(new SoftsignNode()),new ActivationSoftSign(),-1,1);
        testDL4Jnode(initNode(new SwishNode()),new ActivationSwish(),-1,1);
        testDL4Jnode(initNode(new TanhNode()),new ActivationTanH(),-1,1);
        testDL4Jnode(initNode(new HardTanhNode()),new ActivationHardTanH(),-1,1);
        testDL4Jnode(initNode(new RationalTanhNode()),new ActivationRationalTanh(),-1,1);
        testDL4Jnode(initNode(new RectifiedTanhNode()),new ActivationRectifiedTanh(),-1,1);
    }

    @Test
    public void testArithmeticTree() {
        IFunc f = x -> x + (x - (x * (x / (0.1 + DivNode.STABILITY_CONST))));
        IFunc df = x -> 1 + 1 - (2 * x / (0.1 + DivNode.STABILITY_CONST));

        DerivableSymbolicTree tree = (DerivableSymbolicTree) new DerivableSymbolicTree.Builder()
                .setNodeSet(new TreeNodeSet(new Random()))
                .add(new AddNode())
                .add(new InputNode())
                .add(new SubNode())
                .add(new InputNode())
                .add(new MulNode())
                .add(new InputNode())
                .add(new DivNode())
                .add(new InputNode())
                .add(new ConstNode(0.1))
                .build();

        tree = tree.copy();

        testNodeEval((DerivableNode) tree.get(0), f, 1, 2);
        testNodeDeriv((DerivableNode) tree.get(0), df, 1, 2);
    }

    @Test
    public void testTrigonometricTree() {
        IFunc f = x -> Math.sin(Math.cos(Math.tan(x)));
        IFunc df = x -> Math.cos(Math.cos(Math.tan(x))) * (-1) * Math.sin(Math.tan(x)) / Math.cos(x) / Math.cos(x);

        DerivableSymbolicTree tree = (DerivableSymbolicTree) new DerivableSymbolicTree.Builder()
                .setNodeSet(new TreeNodeSet(new Random()))
                .add(new SinNode())
                .add(new CosNode())
                .add(new TanNode())
                .add(new InputNode())
                .build();

        tree = tree.copy();

        testNodeEval((DerivableNode) tree.get(0), f, 1, 2);
        testNodeDeriv((DerivableNode) tree.get(0), df, 1, 2);
    }

    @Test
    public void testExponentialTree() {
        IFunc f = x -> Math.pow(Math.log(Math.pow(Math.exp(Math.pow(x, 2)), 3)), 0.1);
        IFunc df = x -> Math.pow(3 * Math.log(Math.E), 0.1) * 0.2 * Math.pow(x, -0.8);

        DerivableSymbolicTree tree = (DerivableSymbolicTree) new DerivableSymbolicTree.Builder()
                .setNodeSet(new TreeNodeSet(new Random()))
                .add(new PowNode())
                .add(new LogNode())
                .add(new Pow3Node())
                .add(new ExpNode())
                .add(new Pow2Node())
                .add(new InputNode())
                .add(new ConstNode(0.1))
                .build();

        tree = tree.copy();

        testNodeEval((DerivableNode) tree.get(0), f, 1, 2);
        testNodeDeriv((DerivableNode) tree.get(0), df, 1, 2);
    }

    @Test
    public void testActivatedTree() {
        IFunc f = x -> Math.exp(-x * x);
        IFunc df = x -> Math.exp(-x * x) * (-2.) * x;

        DerivableSymbolicTree tree = (DerivableSymbolicTree) new DerivableSymbolicTree.Builder()
                .setNodeSet(new TreeNodeSet(new Random()))
                .add(new GaussNode())
                .add(new InputNode())
                .build();

        tree = tree.copy();

        testNodeEval((DerivableNode) tree.get(0), f, 1, 2);
        testNodeDeriv((DerivableNode) tree.get(0), df, 1, 2);
    }
}