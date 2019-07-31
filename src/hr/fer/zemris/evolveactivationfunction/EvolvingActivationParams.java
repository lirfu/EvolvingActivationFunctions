package hr.fer.zemris.evolveactivationfunction;

import hr.fer.zemris.evolveactivationfunction.nn.NetworkArchitecture;
import hr.fer.zemris.evolveactivationfunction.tree.TreeNodeSets;
import hr.fer.zemris.genetics.Crossover;
import hr.fer.zemris.genetics.Mutation;
import hr.fer.zemris.genetics.stopconditions.StopCondition;
import hr.fer.zemris.neurology.dl4j.TrainParams;
import hr.fer.zemris.utils.IBuilder;
import hr.fer.zemris.utils.ISerializable;
import hr.fer.zemris.utils.Utilities;
import org.jetbrains.annotations.NotNull;

import java.util.Arrays;
import java.util.LinkedList;


public class EvolvingActivationParams extends TrainParams {
    private static ISerializable[] AVAILABLE_OPERATORS;

    private AlgorithmType algorithm_;
    private Integer population_size_;
    private Double mutation_prob_;
    private boolean elitism_;
    private int taboo_size_;
    private int taboo_attempts_;
    private LinkedList<Crossover> crossovers_ = new LinkedList<>();
    private LinkedList<Mutation> mutations_ = new LinkedList<>();
    private StopCondition condition_ = new StopCondition();
    private int worker_num_;
    private long algo_seed_;

    private NetworkArchitecture architecture_;
    private String activation_;
    private String[] node_set_;
    private String train_path_;
    private String test_path_;
    private String experiment_name_;

    static {
        params.put("algorithm", new TrainParamsModifier() {
            @Override
            public Object parse(String s) {
                return AlgorithmType.valueOf(s.toUpperCase());
            }

            @Override
            public void set(TrainParams p, Object value) {
                ((EvolvingActivationParams) p).algorithm_ = (AlgorithmType) value;
            }

            @Override
            public IBuilder<TrainParams> modify(IBuilder<TrainParams> p, Object value) {
                return ((Builder) p).algorithm((AlgorithmType) value);
            }
        });
        params.put("population_size", new TrainParamsModifier() {
            @Override
            public Object parse(String s) {
                return Integer.parseInt(s);
            }

            @Override
            public void set(TrainParams p, Object value) {
                ((EvolvingActivationParams) p).population_size_ = (Integer) value;
            }

            @Override
            public IBuilder<TrainParams> modify(IBuilder<TrainParams> p, Object value) {
                return ((EvolvingActivationParams.Builder) p).population_size((Integer) value);
            }
        });
        params.put("mutation_prob", new TrainParamsModifier() {
            @Override
            public Object parse(String s) {
                return Double.parseDouble(s);
            }

            @Override
            public void set(TrainParams p, Object value) {
                ((EvolvingActivationParams) p).mutation_prob_ = (Double) value;
            }

            @Override
            public IBuilder<TrainParams> modify(IBuilder<TrainParams> p, Object value) {
                return ((EvolvingActivationParams.Builder) p).mutation_prob((Double) value);
            }
        });
        params.put("elitism", new TrainParamsModifier() {
            @Override
            public Object parse(String s) {
                return Boolean.parseBoolean(s);
            }

            @Override
            public void set(TrainParams p, Object value) {
                ((EvolvingActivationParams) p).elitism_ = (boolean) value;
            }

            @Override
            public IBuilder<TrainParams> modify(IBuilder<TrainParams> p, Object value) {
                return ((EvolvingActivationParams.Builder) p).elitism((Boolean) value);
            }
        });
        params.put("taboo_size", new TrainParamsModifier() {
            @Override
            public Object parse(String s) {
                return Integer.parseInt(s);
            }

            @Override
            public void set(TrainParams p, Object value) {
                ((EvolvingActivationParams) p).taboo_size_ = (int) value;
            }

            @Override
            public IBuilder<TrainParams> modify(IBuilder<TrainParams> p, Object value) {
                return ((EvolvingActivationParams.Builder) p).taboo_size((Integer) value);
            }
        });
        params.put("taboo_attempts", new TrainParamsModifier() {
            @Override
            public Object parse(String s) {
                return Integer.parseInt(s);
            }

            @Override
            public void set(TrainParams p, Object value) {
                ((EvolvingActivationParams) p).taboo_attempts_ = (int) value;
            }

            @Override
            public IBuilder<TrainParams> modify(IBuilder<TrainParams> p, Object value) {
                return ((EvolvingActivationParams.Builder) p).taboo_attempts((Integer) value);
            }
        });
        params.put("worker_num", new TrainParamsModifier() {
            @Override
            public Object parse(String s) {
                return Integer.parseInt(s);
            }

            @Override
            public void set(TrainParams p, Object value) {
                ((EvolvingActivationParams) p).worker_num_ = (int) value;
            }

            @Override
            public IBuilder<TrainParams> modify(IBuilder<TrainParams> p, Object value) {
                return ((EvolvingActivationParams.Builder) p).worker_num((Integer) value);
            }
        });
        params.put("architecture", new TrainParamsModifier() {
            @Override
            public Object parse(String s) {
                return new NetworkArchitecture(s);
            }

            @Override
            public void set(TrainParams p, Object value) {
                ((EvolvingActivationParams) p).architecture_ = (NetworkArchitecture) value;
            }

            @Override
            public IBuilder<TrainParams> modify(IBuilder<TrainParams> p, Object value) {
                return ((EvolvingActivationParams.Builder) p).architecture((NetworkArchitecture) value);
            }
        });
        params.put("activation", new TrainParamsModifier() {
            @Override
            public Object parse(String s) {
                return s;
            }

            @Override
            public void set(TrainParams p, Object value) {
                ((EvolvingActivationParams) p).activation_ = (String) value;
            }

            @Override
            public IBuilder<TrainParams> modify(IBuilder<TrainParams> p, Object value) {
                return ((EvolvingActivationParams.Builder) p).activation((String) value);
            }
        });
        params.put("node_set", new TrainParamsModifier() {
            @Override
            public Object parse(String s) {
                return s.split("-");
            }

            @Override
            public void set(TrainParams p, Object value) {
                ((EvolvingActivationParams) p).node_set_ = (String[]) value;
            }

            @Override
            public IBuilder<TrainParams> modify(IBuilder<TrainParams> p, Object value) {
                EvolvingActivationParams.Builder par = ((EvolvingActivationParams.Builder) p);
                par.node_set_ = new LinkedList<>();
                par.node_set_.addAll(Arrays.asList((String[]) value));
                return par;
            }
        });
        params.put("train_path", new TrainParamsModifier() {
            @Override
            public Object parse(String s) {
                return s;
            }

            @Override
            public void set(TrainParams p, Object value) {
                ((EvolvingActivationParams) p).train_path_ = (String) value;
            }

            @Override
            public IBuilder<TrainParams> modify(IBuilder<TrainParams> p, Object value) {
                return null;
            }
        });
        params.put("test_path", new TrainParamsModifier() {
            @Override
            public Object parse(String s) {
                return s;
            }

            @Override
            public void set(TrainParams p, Object value) {
                ((EvolvingActivationParams) p).test_path_ = (String) value;
            }

            @Override
            public IBuilder<TrainParams> modify(IBuilder<TrainParams> p, Object value) {
                return null;
            }
        });
        params.put("algo_seed", new TrainParamsModifier() {
            @Override
            public Object parse(String s) {
                return Long.parseLong(s);
            }

            @Override
            public void set(TrainParams p, Object value) {
                ((EvolvingActivationParams) p).algo_seed_ = (Long) value;
            }

            @Override
            public IBuilder<TrainParams> modify(IBuilder<TrainParams> p, Object value) {
                return ((EvolvingActivationParams.Builder) p).algo_seed((Long) value);
            }
        });
    }

    public EvolvingActivationParams() {
    }

    public EvolvingActivationParams(TrainParams train_params, AlgorithmType algorithm, int population_size, double mutation_prob,
                                    boolean elitism, int taboo_size, int taboo_attempts,
                                    LinkedList<Crossover> crossovers, LinkedList<Mutation> mutations,
                                    StopCondition condition, int worker_num,
                                    NetworkArchitecture architecture, String activation, String[] node_set,
                                    String train_path, String test_path, String experiment_name, long algo_seed) {
        super(train_params);
        algorithm_ = algorithm;
        population_size_ = population_size;
        mutation_prob_ = mutation_prob;
        elitism_ = elitism;
        taboo_size_ = taboo_size;
        taboo_attempts_ = taboo_attempts;
        crossovers_ = crossovers;
        mutations_ = mutations;
        condition_ = condition;
        worker_num_ = worker_num;
        architecture_ = architecture;
        activation_ = activation;
        node_set_ = node_set;
        train_path_ = train_path;
        test_path_ = test_path;
        experiment_name_ = experiment_name;
        algo_seed_ = algo_seed;
    }

    public static void initialize(ISerializable[] available_nodes) {
        AVAILABLE_OPERATORS = available_nodes;
    }

    public AlgorithmType getAlgorithm() {
        return algorithm_;
    }

    public int population_size() {
        return population_size_;
    }

    public double mutation_prob() {
        return mutation_prob_;
    }

    public boolean isElitism() {
        return elitism_;
    }

    public int taboo_size() {
        return taboo_size_;
    }

    public int taboo_attempts() {
        return taboo_attempts_;
    }

    public LinkedList<Crossover> crossovers() {
        return crossovers_;
    }

    public LinkedList<Mutation> mutations() {
        return mutations_;
    }

    public StopCondition condition() {
        return condition_;
    }

    public int worker_num() {
        return worker_num_;
    }

    public NetworkArchitecture architecture() {
        return architecture_;
    }

    public String activation() {
        return activation_;
    }

    public void activation(String activation) {
        activation_ = activation;
    }

    public String[] node_set() {
        return node_set_;
    }

    public String train_path() {
        return train_path_;
    }

    public void train_path(String path) {
        train_path_ = path;
    }

    public String test_path() {
        return test_path_;
    }

    public void test_path(String path) {
        test_path_ = path;
    }

    public String experiment_name() {
        return experiment_name_;
    }

    public void experiment_name(String name) {
        experiment_name_ = name;
    }

    public Long algo_seed() {
        return algo_seed_;
    }

    @Override
    public String serialize() {
        StringBuilder sb = new StringBuilder("# NN params\n" + super.serialize());
        if (architecture_ != null) {
            sb.append("architecture").append('\t').append(architecture_.serialize()).append('\n');
        }
        if (activation_ != null)
            sb.append("activation").append('\t').append(activation_).append('\n');
        sb.append("\n# GA params\n")
                .append("algo_seed").append('\t').append(algo_seed_).append('\n')
                .append("algorithm").append('\t').append(algorithm_).append('\n')
                .append("population_size").append('\t').append(population_size_).append('\n')
                .append("mutation_prob").append('\t').append(mutation_prob_).append('\n')
                .append("elitism").append('\t').append(elitism_).append('\n')
                .append("taboo_size").append('\t').append(taboo_size_).append('\n')
                .append("taboo_attempts").append('\t').append(taboo_attempts_).append('\n')
                .append("worker_num").append('\t').append(worker_num_).append('\n')
                .append("node_set").append('\t').append(String.join("-", node_set_)).append('\n');
        if (condition_ != null) {
            sb.append("\n# Stop conditions:\n");
            sb.append(condition_.serialize());
        }
        sb.append("\n# GA operators\n");
        if (crossovers_ != null)
            for (Crossover c : crossovers_)
                sb.append(c.serialize());
        if (mutations_ != null)
            for (Mutation m : mutations_)
                sb.append(m.serialize());
        sb.append("\n# Dataset\n");
        sb.append("train_path").append('\t').append(train_path_).append('\n');
        if (test_path_ != null)
            sb.append("test_path").append('\t').append(test_path_).append('\n');
        sb.append("experiment_name").append('\t').append(experiment_name_).append('\n');
        return sb.toString();
    }

    @Override
    public boolean parse(String line) {
        if (super.parse(line)) return true;

        String[] parts = line.split(Utilities.KEY_VALUE_SIMPLE_REGEX);
        if (parts.length == 2 && parts[0].equals("experiment_name")) {
            experiment_name_ = parts[1];
            return true;
        }

        if (condition_.parse(line)) return true;

        if (AVAILABLE_OPERATORS == null) {
            throw new IllegalStateException("EvolvingActivationParams wasn't initialized! Please call the static initialize() method before parsing.");
        } else {
            for (ISerializable o : AVAILABLE_OPERATORS) {
                if (o.parse(line)) {
                    if (o instanceof Crossover && !crossovers_.contains(o)) {
                        crossovers_.add((Crossover) o);
                        return true;
                    } else if (o instanceof Mutation && !mutations_.contains(o)) {
                        mutations_.add((Mutation) o);
                        return true;
                    }
                }
            }
        }

        return false;
    }

    public enum AlgorithmType {
        GENERATION, GENERATION_TABOO, ELIMINATION
    }

    public static class Builder extends TrainParams.Builder {
        private AlgorithmType algorithm_ = AlgorithmType.GENERATION_TABOO;
        private int population_size_;
        private double mutation_prob_ = 0.1;
        private boolean elitism_ = true;
        private int taboo_size_;
        private int taboo_attempts_;
        private LinkedList<Crossover> crossovers_ = new LinkedList<>();
        private LinkedList<Mutation> mutations_ = new LinkedList<>();
        private StopCondition condition_;
        private NetworkArchitecture architecture_;
        private String activation_;
        private LinkedList<String> node_set_ = new LinkedList<>();
        private String train_path_, test_path_;
        private String experiment_name_;
        private int worker_num_ = 1;
        private long algo_seed_ = 42;

        public EvolvingActivationParams build() {
            if (experiment_name_ == null)
                throw new IllegalStateException("Experiment name must be defined!");
            if (architecture_ == null)
                throw new IllegalStateException("Network architecture must be defined!");
            if (train_path_ == null)
                throw new IllegalStateException("Train dataset path must be specified!");
            if (condition_ == null)
                throw new IllegalStateException("Stop condition must be specified!");
            if (node_set_.isEmpty())
                throw new IllegalStateException("Node set must be specified!");

            return new EvolvingActivationParams(super.build(), algorithm_, population_size_, mutation_prob_,
                    elitism_, taboo_size_, taboo_attempts_, crossovers_, mutations_, condition_, worker_num_,
                    architecture_, activation_, node_set_.toArray(new String[]{}),
                    train_path_, test_path_, experiment_name_, algo_seed_);
        }

        public Builder algorithm(AlgorithmType algorithm) {
            algorithm_ = algorithm;
            return this;
        }

        public Builder population_size(int size) {
            population_size_ = size;
            return this;
        }

        public Builder mutation_prob(double probability) {
            mutation_prob_ = probability;
            return this;
        }

        public Builder stop_condition(StopCondition condtion) {
            condition_ = condtion;
            return this;
        }

        public Builder elitism(boolean elitism) {
            elitism_ = elitism;
            return this;
        }

        public Builder taboo_size(int size) {
            taboo_size_ = size;
            return this;
        }

        public Builder taboo_attempts(int num) {
            taboo_attempts_ = num;
            return this;
        }

        public Builder addCrossover(Crossover c) {
            crossovers_.add(c);
            return this;
        }

        public Builder addMutation(Mutation m) {
            mutations_.add(m);
            return this;
        }

        public Builder architecture(NetworkArchitecture arch) {
            architecture_ = arch;
            return this;
        }

        public Builder activation(String activation) {
            activation_ = activation;
            return this;
        }

        public Builder addNodeSet(String set_name) {
            TreeNodeSets.valueOf(set_name);
            node_set_.add(set_name);
            return this;
        }

        public Builder train_path(String path) {
            train_path_ = path;
            return this;
        }

        public Builder test_path(String path) {
            test_path_ = path;
            return this;
        }

        public Builder experiment_name(String experiment_name) {
            experiment_name_ = experiment_name;
            return this;
        }

        public Builder worker_num(int num) {
            worker_num_ = num;
            return this;
        }

        public Builder algo_seed(long seed) {
            algo_seed_ = seed;
            return this;
        }

        public Builder cloneFrom(@NotNull EvolvingActivationParams p) {
            super.cloneFrom(p);
            algorithm_ = p.algorithm_;
            population_size_ = p.population_size_;
            mutation_prob_ = p.mutation_prob_;
            elitism_ = p.elitism_;
            taboo_size_ = p.taboo_size_;
            taboo_attempts_ = p.taboo_attempts_;
            crossovers_ = new LinkedList<>(p.crossovers_);
            mutations_ = new LinkedList<>(p.mutations_);
            condition_ = p.condition_;
            architecture_ = p.architecture_;
            activation_ = p.activation_;

            node_set_ = new LinkedList<>();
            node_set_.addAll(Arrays.asList(p.node_set_));

            train_path_ = p.train_path_;
            test_path_ = p.test_path_;
            experiment_name_ = p.experiment_name_;
            worker_num_ = p.worker_num_;
            algo_seed_ = p.algo_seed_;
            return this;
        }
    }
}
