package hr.fer.zemris.genetics.symboregression;

import hr.fer.zemris.evolveactivationfunction.TreeNodeSetFactory;
import hr.fer.zemris.genetics.*;
import hr.fer.zemris.genetics.algorithms.GenerationTabooAlgorithm;
import hr.fer.zemris.genetics.selectors.RouletteWheelSelector;
import hr.fer.zemris.genetics.stopconditions.StopCondition;
import hr.fer.zemris.genetics.symboregression.crx.CrxSRSwapConstants;
import hr.fer.zemris.genetics.symboregression.crx.CrxSRSwapNodes;
import hr.fer.zemris.genetics.symboregression.crx.CrxSRSwapSubtrees;
import hr.fer.zemris.genetics.symboregression.mut.*;
import hr.fer.zemris.genetics.symboregression.nodes.ConstNode;
import hr.fer.zemris.utils.Pair;
import hr.fer.zemris.utils.Triple;
import hr.fer.zemris.utils.logs.ILogger;
import hr.fer.zemris.utils.logs.LogLevel;
import hr.fer.zemris.utils.logs.StdoutLogger;

import java.util.Random;

public class SymbolicRegressionDemo {
    private static final long SEED = 12;

    public static double func(State s) {
        return Math.max(s.x_ * s.y_, s.y_ * Math.exp(Math.sin(s.z_ * 5)));
    }

    public static void main(String[] args) {
        Random rand = new Random(SEED);
        ILogger logger = new StdoutLogger(new LogLevel(LogLevel.DEBUG));

        TreeNodeSet node_set = new TreeNodeSet(rand);
        node_set.registerBinaryOperator(new AddNode());
        node_set.registerBinaryOperator(new MulNode());
        node_set.registerBinaryOperator(new MaxNode());
        node_set.registerBinaryOperator(new MinNode());
//        node_set.registerBinaryOperator(new DivNode());
        node_set.registerUnaryOperator(new ExpNode());
        node_set.registerUnaryOperator(new SinNode());
        node_set.registerTerminal(new XNode());
        node_set.registerTerminal(new YNode());
        node_set.registerTerminal(new ZNode());
        node_set.registerTerminal(new ConstNode<State>(1.));

        StopCondition cond = new StopCondition.Builder()
                .setMaxIterations(50)
                .setMinFitness(0)
                .build();

        GenerationTabooAlgorithm algo = (GenerationTabooAlgorithm) new GenerationTabooAlgorithm.Builder()
                // Algorithm specific params.
                .setElitism(true)
                .setTabooSize(60)
                .setTabooAttempts(30)
                // Common params.
                .setRandom(rand)
                .setLogger(logger)
                .setPopulationSize(50)
                .setMutationProbability(0.3)
                .setTopOptimaNumber(5)
                .setStopCondition(cond)
                .setNumberOfWorkers(4)
                // Problem specific params.
                .setGenotypeTemplate(new SymbolicTree<>(node_set, null))
                .setSelector(new RouletteWheelSelector(rand))
                .setInitializer(new SRGenericInitializer(node_set, 4))
                .setEvaluator(new Eval())
                // Tree crossovers.
                .addCrossover(new CrxSRSwapSubtrees(rand).setImportance(1))
                .addCrossover(new CrxSRSwapNodes(rand).setImportance(1))
                .addCrossover((Crossover) new CrxReturnRandom(rand).setImportance(1))
                // Tree mutations.
                .addMutation(new MutSRSwapOrder(rand).setImportance(1))
                .addMutation(new MutSRInsertTerminal(node_set, rand).setImportance(1))
                .addMutation(new MutSRInsertRoot(node_set, rand).setImportance(1))
                .addMutation(new MutSRReplaceNode(node_set, rand).setImportance(1))
                .addMutation(new MutSRReplaceSubtree(node_set, new SRGenericInitializer(node_set, 3), rand).setImportance(1))
                .addMutation(new MutSRRemoveRoot(rand).setImportance(1))
                .addMutation(new MutSRRemoveUnary(rand))
//                .addMutation(new MutInitialize<>(new SRGenericInitializer(node_set, 2)).setImportance(1))
                // Constants crossovers.
                .addCrossover(new CrxSRSwapConstants(rand).setImportance(1))
//                .addCrossover(new CrxSRMeanConstants(rand).setImportance(2))
                // Constants mutations.
                .addMutation(new MutSRRandomConstantSet(rand, -10, 10).setImportance(1))
                .addMutation(new MutSRRandomConstantAdd(rand, 2).setImportance(1))
//                .addMutation(new MutSRRandomConstantSetInt(rand, 0, 10).setImportance(3))
                .build();

//        algo.run();
        algo.run(new Algorithm.LogParams(false, true));

        SymbolicTree best = (SymbolicTree) algo.getBest();
        System.out.println("Final result: " + best + "   (" + best.getFitness() + ")");

        System.out.println("Top 5 results:");
        for (Triple<Long, String, Double> g : algo.getResultBundle().getOptimumHistory()) {
            System.out.println(g.getKey() + ". " + g.getVal() + "   (" + g.getExtra() + ")");
        }

        System.out.println("Done!");
    }

    public static class Eval extends AEvaluator<SymbolicTree<State, Double>> {
        private final int min_ = -5, max_ = 5;

        @Override
        public double performEvaluate(SymbolicTree<State, Double> g) {
            State s = new State(0, 0, 0);
            double fitness = 0;
            for (s.x_ = min_; s.x_ <= max_; s.x_++) {
                for (s.y_ = min_ - 8; s.y_ <= max_ + 2; s.y_++) {
                    for (s.z_ = min_ - 4; s.z_ <= max_ + 3; s.z_++) {
                        double true_val = func(s);
                        double pred_val = g.execute(s);
                        fitness += Math.pow(true_val - pred_val, 2);
                        if (!Double.isFinite(fitness)) {
                            return Double.MAX_VALUE;
                        }
                    }
                }
            }
            return fitness / Math.pow(max_ - min_, 3);
        }
    }

    public static class State {
        public double x_;
        public double y_;
        public double z_;

        public State(double x, double y, double z) {
            x_ = x;
            y_ = y;
            z_ = z;
        }
    }

    public static class AddNode extends TreeNode<State, Double> {
        AddNode() {
            super("+", 2);
        }

        @Override
        protected IExecutable<State, Double> getExecutable() {
            return (state, n) -> ((Double) n.getChild(0).execute(state)) + ((Double) n.getChild(1).execute(state));
        }

        @Override
        protected IInstantiable<TreeNode<State, Double>> getInstantiable() {
            return AddNode::new;
        }
    }

    public static class MulNode extends TreeNode<State, Double> {
        MulNode() {
            super("*", 2);
        }

        @Override
        protected IExecutable<State, Double> getExecutable() {
            return (state, n) -> ((Double) n.getChild(0).execute(state)) * ((Double) n.getChild(1).execute(state));
        }

        @Override
        protected IInstantiable<TreeNode<State, Double>> getInstantiable() {
            return MulNode::new;
        }
    }

    public static class DivNode extends TreeNode<State, Double> {
        DivNode() {
            super("/", 2);
        }

        @Override
        protected IExecutable<State, Double> getExecutable() {
            return (state, n) -> ((Double) n.getChild(0).execute(state)) / ((Double) n.getChild(1).execute(state));
        }

        @Override
        protected IInstantiable<TreeNode<State, Double>> getInstantiable() {
            return DivNode::new;
        }
    }

    public static class MaxNode extends TreeNode<State, Double> {
        MaxNode() {
            super("max", 2);
        }

        @Override
        protected IExecutable<State, Double> getExecutable() {
            return (state, n) -> Math.max((Double) n.getChild(0).execute(state), (Double) n.getChild(1).execute(state));
        }

        @Override
        protected IInstantiable<TreeNode<State, Double>> getInstantiable() {
            return MaxNode::new;
        }
    }

    public static class MinNode extends TreeNode<State, Double> {
        MinNode() {
            super("min", 2);
        }

        @Override
        protected IExecutable<State, Double> getExecutable() {
            return (state, n) -> Math.min((Double) n.getChild(0).execute(state), (Double) n.getChild(1).execute(state));
        }

        @Override
        protected IInstantiable<TreeNode<State, Double>> getInstantiable() {
            return MinNode::new;
        }
    }

    public static class ExpNode extends TreeNode<State, Double> {
        ExpNode() {
            super("exp", 1);
        }

        @Override
        protected IExecutable<State, Double> getExecutable() {
            return (state, n) -> Math.exp((Double) n.getChild(0).execute(state));
        }

        @Override
        protected IInstantiable<TreeNode<State, Double>> getInstantiable() {
            return ExpNode::new;
        }
    }

    public static class SinNode extends TreeNode<State, Double> {
        SinNode() {
            super("sin", 1);
        }

        @Override
        protected IExecutable<State, Double> getExecutable() {
            return (state, n) -> Math.sin((Double) n.getChild(0).execute(state));
        }

        @Override
        protected IInstantiable<TreeNode<State, Double>> getInstantiable() {
            return SinNode::new;
        }
    }

    public static class XNode extends TreeNode<State, Double> {
        XNode() {
            super("x", 0);
        }

        @Override
        protected IExecutable<State, Double> getExecutable() {
            return (state, n) -> state.x_;
        }

        @Override
        protected IInstantiable<TreeNode<State, Double>> getInstantiable() {
            return XNode::new;
        }
    }

    public static class YNode extends TreeNode<State, Double> {
        YNode() {
            super("y", 0);
        }

        @Override
        protected IExecutable<State, Double> getExecutable() {
            return (state, n) -> state.y_;
        }

        @Override
        protected IInstantiable<TreeNode<State, Double>> getInstantiable() {
            return YNode::new;
        }
    }

    public static class ZNode extends TreeNode<State, Double> {

        ZNode() {
            super("z", 0);
        }

        @Override
        protected IExecutable<State, Double> getExecutable() {
            return (state, n) -> state.z_;
        }

        @Override
        protected IInstantiable<TreeNode<State, Double>> getInstantiable() {
            return ZNode::new;
        }

    }
}
