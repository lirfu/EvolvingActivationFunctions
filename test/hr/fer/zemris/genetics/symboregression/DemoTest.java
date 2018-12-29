package hr.fer.zemris.genetics.symboregression;

import hr.fer.zemris.genetics.AEvaluator;
import org.junit.Test;

import java.util.Random;

import static org.junit.Assert.*;

public class DemoTest {
    private static AEvaluator<SymbolicTree<Demo.State, Double>> eval_ = new AEvaluator<SymbolicTree<Demo.State, Double>>() {
        private final int min_ = -100, max_ = 100;

        @Override
        public double performEvaluate(SymbolicTree<Demo.State, Double> g) {
            double fitness = 0;
            Demo.State state = new Demo.State(0, 0);
            for (state.x_ = min_; state.x_ <= max_; state.x_++) {
                for (state.y_ = min_; state.y_ <= max_; state.y_++) {
                    double true_val = state.x_ * state.y_ + state.x_;
                    double pred_val = g.execute(state);
                    fitness += Math.pow(true_val - pred_val, 2);
                }
            }
            fitness /= Math.pow(max_ - min_ + 1, 2);
            if (!Double.isFinite(fitness)) {
                return Double.MAX_VALUE;
            }
            return fitness;
        }

    };

    private static SymbolicTree buildSolution() {
        return new SymbolicTree.Builder()
                .setNodeSet(new TreeNodeSet(new Random()))
                .add(new Demo.AddNode())
                .add(new Demo.MulNode())
                .add(new Demo.XNode())
                .add(new Demo.YNode())
                .add(new Demo.XNode())
                .build();
    }

    @Test
    public void testGetSet() {
        SymbolicTree t = buildSolution();

        assertTrue("Getter should work correctly.", t.get(4).equals(new Demo.XNode()));

        t.set(3, new Demo.XNode());

        assertTrue("Setter should work correctly.", t.get(3).equals(new Demo.XNode()));
    }

    @Test
    public void testOptimal() {
        SymbolicTree t = buildSolution();
        double val = eval_.performEvaluate(t);
        assertTrue("Optimal function must satisfy evaluator: " + val, val == 0);
    }

    @Test
    public void testSubOptimal() {
        SymbolicTree t = buildSolution();
        TreeNode[] nodes = new TreeNode[]{new Demo.XNode(), new Demo.YNode()};

        for (int i = 0; i < t.size(); i++) {
            TreeNode orig = t.get(i).clone();

            for (TreeNode n : nodes) {
                if (orig.equals(n)) continue; // Skip original nodes.

                t.set(i, n.clone());

                assertTrue("Non-optimal function must not satisfy evaluator: " + t, eval_.performEvaluate(t) > 0);
            }

            t.set(i, orig.clone());
        }
    }

    @Test
    public void testCloning() {
        Demo.State s = new Demo.State(3, 5);
        SymbolicTree t = buildSolution();
        SymbolicTree c = t.copy();

        TreeNode n = new Demo.AddNode();
        n.children_[0] = new Demo.XNode();
        n.children_[1] = new Demo.XNode();

        assertTrue("Node should clone properly.", n.clone().equals(n));

        c.set(3, n.clone());

        assertFalse("Modifying a cloned tree mustn't affect the original.", t.get(3).equals(c.get(3)));

        assertTrue("Modified node should execute as expected.", c.get(3).execute(s).equals(n.execute(s)));

        assertFalse("Modified tree should execute differently from original.", t.execute(s) == c.execute(s));

    }
}