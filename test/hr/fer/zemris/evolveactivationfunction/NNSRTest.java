package hr.fer.zemris.evolveactivationfunction;

import hr.fer.zemris.evolveactivationfunction.activationfunction.DerivableNode;
import hr.fer.zemris.evolveactivationfunction.nodes.*;
import hr.fer.zemris.genetics.symboregression.IExecutable;
import hr.fer.zemris.genetics.symboregression.SymbolicTree;
import hr.fer.zemris.genetics.symboregression.TreeNode;
import hr.fer.zemris.genetics.symboregression.TreeNodeSet;
import org.junit.Test;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Random;

import static org.junit.Assert.assertTrue;


public class NNSRTest {
    @Test
    public void testParsing() {
        TreeNodeSetFactory factory = new TreeNodeSetFactory();
        TreeNodeSet set = factory.build(new Random(42), TreeNodeSetFactory.Set.ARITHMETICS, TreeNodeSetFactory.Set.TRIGONOMETRY, TreeNodeSetFactory.Set.CONSTANT);

        String string = "+[x,*[x,sin[*[-273.15,x]]]]";
        SymbolicTree tree = SymbolicTree.parse(string, set);

        assertTrue("Serialized tree must equal the inputted parse string.", tree.toString().equals(string));
    }

    private interface IFunc {
        public double f(double x);
    }

    private final double REQUIRED_PRECISION = 1e-12;

    private void testNode(DerivableNode n, IFunc f, double start, double end) {
        assertTrue("Test boundaries don't satisfy: " + start + "<" + end, start < end);

        for (double x = start; x <= end; x += 0.01) {
            INDArray input = Nd4j.scalar(x);
            double p = n.execute(input).getDouble(0);
            double t = f.f(x);
            assertTrue("[" + n.getName() + "] should calculate correctly: " + p + "!=" + t,
                    Math.abs(t - p) <= REQUIRED_PRECISION);
            assertTrue("[" + n.getName() + "] must not change the input matrix: " + x + "!=" + input.getDouble(0),
                    Math.abs(x - input.getDouble(0)) <= REQUIRED_PRECISION);
        }
    }

    private DerivableNode initNode(DerivableNode n) {
        for (int i = 0; i < n.getChildrenNum(); i++) {
            n.getChildren()[i] = new InputNode();
        }
        return n;
    }

    @Test
    public void testNodes() {
        // Must set precision to double.
        Nd4j.setDataType(DataBuffer.Type.DOUBLE);

        // Input node.
        testNode(new InputNode(), x -> x, -1, 1);
        // Arithmetic.
        testNode(initNode(new AddNode()), x -> x + x, -1, 1);
        testNode(initNode(new SubNode()), x -> x - x, -1, 1);
        testNode(initNode(new MulNode()), x -> x * x, -1, 1);
        testNode(initNode(new DivNode()), x -> x / (x + DivNode.STABILITY_CONST), 1, 3);
        // Trigonometry.
        testNode(initNode(new SinNode()), Math::sin, -1, 1);
        testNode(initNode(new CosNode()), Math::cos, -1, 1);
        testNode(initNode(new TanNode()), Math::tan, -1, 1);
        // Exponentials.
        testNode(initNode(new ExpNode()), Math::exp, -1, 1);
        testNode(initNode(new Pow2Node()), x -> Math.pow(x, 2), -1, 1);
        testNode(initNode(new Pow3Node()), x -> Math.pow(x, 3), -1, 1);
        testNode(initNode(new PowNode()), x -> Math.pow(x, x), 0, 2);
        testNode(initNode(new LogNode()), Math::log, 1e-3, 2);
        testNode(initNode(new GaussNode()), x -> Math.exp(-x * x), -3, 3);
        // Constant.
        ConstNode n = new ConstNode();
        n.setExtra(-273.15);
        testNode(n, x -> -273.15, -0.1, 0.1);
        // Activations.
        testNode(initNode(new ReLUNode()), x -> Math.max(0, x), -1, 1);
        testNode(initNode(new SigmoidNode()), x -> 1. / (1 + Math.exp(-x)), -1, 1);
    }
}