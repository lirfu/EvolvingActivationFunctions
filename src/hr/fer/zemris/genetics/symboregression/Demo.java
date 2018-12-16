package hr.fer.zemris.genetics.symboregression;

import hr.fer.zemris.genetics.AEvaluator;
import hr.fer.zemris.genetics.Crossover;
import hr.fer.zemris.genetics.Mutation;
import hr.fer.zemris.genetics.algorithms.GenerationAlgorithm;
import hr.fer.zemris.genetics.selectors.RouletteWheelSelector;
import hr.fer.zemris.utils.logs.ILogger;
import hr.fer.zemris.utils.logs.StdoutLogger;

import java.util.Random;

public class Demo {
    private static class Eval extends AEvaluator<SymbolicTree<State>> {
        public static double func(double x, double y, double z) {
            return x * y + z;
        }

        @Override
        public double performEvaluate(SymbolicTree<State> g) {
            double fitness = 0;
            for (int x = 0; x < 2; x++) {
                for (int y = 0; y < 2; y++) {
                    for (int z = 0; z < 2; z++) {
                        double true_val = func(x, y, z);
                        double pred_val = (double) g.execute(new State(x, y, z));
                        fitness += 1. / (1 + Math.pow(true_val - pred_val, 2));
                    }
                }
            }
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
        protected AddNode() {
            super("+", 2);
        }

        @Override
        public Double execute(State state) {
            return ((Double) children_[0].execute(state)) + ((Double) children_[1].execute(state));
        }

        @Override
        public AddNode getInstance() {
            return new AddNode();
        }
    }

    public static class MulNode extends TreeNode<State, Double> {
        protected MulNode() {
            super("*", 2);
        }

        @Override
        public Double execute(State state) {
            return ((Double) children_[0].execute(state)) * ((Double) children_[1].execute(state));
        }

        @Override
        public MulNode getInstance() {
            return new MulNode();
        }
    }

    public static class XNode extends TreeNode<State, Double> {
        protected XNode() {
            super("x", 0);
        }

        @Override
        public Double execute(State state) {
            return state.x_;
        }

        @Override
        public XNode getInstance() {
            return new XNode();
        }
    }

    public static class YNode extends TreeNode<State, Double> {
        protected YNode() {
            super("y", 0);
        }

        @Override
        public Double execute(State state) {
            return state.y_;
        }

        @Override
        public YNode getInstance() {
            return new YNode();
        }
    }

    public static class ZNode extends TreeNode<State, Double> {
        protected ZNode() {
            super("z", 0);
        }

        @Override
        public Double execute(State state) {
            return state.z_;
        }

        @Override
        public ZNode getInstance() {
            return new ZNode();
        }
    }

    public static void main(String[] args) {
        Random rand = new Random(42);
        ILogger logger = new StdoutLogger();

        TreeNodeSet factory = new TreeNodeSet();
        factory.registerOperatorNode(new AddNode());
        factory.registerOperatorNode(new MulNode());
        factory.registerTerminalNode(new XNode());
        factory.registerTerminalNode(new YNode());
        factory.registerTerminalNode(new ZNode());

        GenerationAlgorithm algo = new GenerationAlgorithm(new GenerationAlgorithm.Builder()
                .setSeed(42)
                .setLogger(logger)
                .setPopulationSize(10)
                .setMaxIterationsCondition(10)
                .setGenotypeTemplate(new SymbolicTree(factory))
                .setSelector(new RouletteWheelSelector(rand))
                .setEvaluator(new Eval())
                .addCrossover(new Crossover<SymbolicTree<State>>() { // Swap two random nodes.
                    @Override
                    public SymbolicTree<State> cross(SymbolicTree<State> parent1, SymbolicTree<State> parent2) {
                        SymbolicTree<State> child1 = parent1.copy();
                        SymbolicTree<State> child2 = parent2.copy();
                        TreeNode n1 = child1.get(1 + rand.nextInt(child1.size() - 1));
                        TreeNode n2 = child2.get(1 + rand.nextInt(child2.size() - 1));
                        TreeNode.swapContents(n1, n2);
                        return rand.nextBoolean() ? child1 : child2;
                    }
                })
                .addMutation(new Mutation<SymbolicTree<State>>() { // Replace random child with a terminal.
                    @Override
                    public void mutate(SymbolicTree<State> genotype) {
                        TreeNode n = genotype.get(rand.nextInt(genotype.size()));
                        if (n.getChildrenNum() > 0)
                            n.getChildren()[rand.nextInt(n.getChildrenNum())] = factory.getRandomTerminal(rand);
                    }
                })
                .addMutation(new Mutation<SymbolicTree<State>>() { // Add random operator with terminals.
                    @Override
                    public void mutate(SymbolicTree<State> genotype) {
                        TreeNode n = genotype.get(rand.nextInt(genotype.size()));
                        // Build a random operator.
                        TreeNode node = factory.getRandomOperator(rand);
                        for (int i = 0; i < node.getChildrenNum(); i++) {
                            node.getChildren()[i] = factory.getRandomTerminal(rand);
                        }
                        n.getChildren()[rand.nextInt(n.getChildrenNum())] = node;
                    }
                })
                .setNumberOfWorkers(1)
                .build(), true);

        algo.run();

        System.out.println("Final result:");
        System.out.println(algo.getResultBundle());
    }
}
