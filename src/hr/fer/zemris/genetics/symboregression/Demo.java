package hr.fer.zemris.genetics.symboregression;

import hr.fer.zemris.genetics.*;
import hr.fer.zemris.genetics.algorithms.GenerationAlgorithm;
import hr.fer.zemris.genetics.algorithms.GenerationTabooAlgorithm;
import hr.fer.zemris.genetics.selectors.RouletteWheelSelector;
import hr.fer.zemris.genetics.stopconditions.StopCondition;
import hr.fer.zemris.utils.logs.ILogger;
import hr.fer.zemris.utils.logs.LogLevel;
import hr.fer.zemris.utils.logs.StdoutLogger;

import java.util.Random;

public class Demo {
    private static final long SEED = 12;

    public static double func(State s) {
        return s.x_ * s.y_ + s.z_;
    }

    public static void main(String[] args) {
        Random rand = new Random(SEED);
        ILogger logger = new StdoutLogger(new LogLevel(LogLevel.DEBUG));

        TreeNodeSet node_set = new TreeNodeSet(rand);
        node_set.registerBinaryOperator(new AddNode());
        node_set.registerBinaryOperator(new MulNode());
//        node_set.registerBinaryOperator(new DivNode());
//        node_set.registerUnaryOperator(new ExpNode());
        node_set.registerTerminal(new XNode());
        node_set.registerTerminal(new YNode());
        node_set.registerTerminal(new ZNode());

        StopCondition cond = new StopCondition.Builder()
                .setMaxIterations(20)
                .setMinFitness(0)
                .build();

        GenerationTabooAlgorithm algo = (GenerationTabooAlgorithm) new GenerationTabooAlgorithm.Builder()
                .setElitism(true)
                .setTabooSize(15)
                .setTabooAttempts(5)

                .setRandom(rand)
                .setLogger(logger)
                .setPopulationSize(10)
                .setMutationProbability(0.4)
                .setStopCondition(cond)

                .setGenotypeTemplate(new SymbolicTree<>(node_set, null))
                .setSelector(new RouletteWheelSelector(rand))
                .setInitializer(new SRGenericInitializer(node_set, 4))
                .setEvaluator(new Eval())

                .addCrossover(new CrxSRSwapSubtree(rand).setImportance(2))
                .addCrossover((Crossover) new CrxRandom(rand).setImportance(1))

                .addMutation(new MutSRSwapOrder(rand).setImportance(1))
                .addMutation(new MutSRInsertTerminal(node_set, rand).setImportance(3))
                .addMutation(new MutSRInsertRoot(node_set, rand).setImportance(2))
                .addMutation(new MutSRReplaceNode(node_set, rand).setImportance(2))
                .addMutation(new MutSRReplaceSubtree(node_set, new SRGenericInitializer(node_set, 2), rand).setImportance(2))

                .setNumberOfWorkers(7)
                .build();

//        algo.run();
        algo.run(new Algorithm.LogParams(false, true));

        Genotype best = algo.getBest();
        System.out.println(best);

        System.out.println("Done!");
    }

    public static class Eval extends AEvaluator<SymbolicTree<State, Double>> {
        private final int min_ = -5, max_ = 5;
        private final State state_ = new State(-1, -1, -1);

        @Override
        public double performEvaluate(SymbolicTree<State, Double> g) {
            double fitness = 0;
            for (state_.x_ = min_; state_.x_ <= max_; state_.x_++) {
                for (state_.y_ = min_; state_.y_ <= max_; state_.y_++) {
                    for (state_.z_ = min_; state_.z_ <= max_; state_.z_++) {
                        double true_val = func(state_);
                        double pred_val = g.execute(state_);
                        fitness += Math.pow(true_val - pred_val, 2);
                    }
                }
            }
//            if (Double.isNaN(fitness)) {
//                return Double.MAX_VALUE;
//            }
            return fitness;
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
            return () -> new AddNode();
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
            return () -> new MulNode();
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
            return () -> new DivNode();
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
            return () -> new ExpNode();
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
            return () -> new XNode();
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
            return () -> new YNode();
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
            return () -> new ZNode();
        }
    }
}
