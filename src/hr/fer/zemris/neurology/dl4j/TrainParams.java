package hr.fer.zemris.neurology.dl4j;

import hr.fer.zemris.experiments.GridSearch;
import hr.fer.zemris.utils.IBuilder;
import hr.fer.zemris.utils.ISerializable;
import hr.fer.zemris.utils.Pair;
import hr.fer.zemris.utils.Utilities;
import org.jetbrains.annotations.NotNull;

import java.util.HashMap;
import java.util.LinkedList;
import java.util.regex.Matcher;

/**
 * Immutable wrapper for train parameters.
 */
public class TrainParams implements ISerializable {
    private int input_size_, output_size_;
    private int epochs_num_, batch_size_;
    private boolean normalize_features_, shuffle_batches_, batch_norm_;
    private double learning_rate_, decay_rate_;
    private int decay_step_;
    private double regularization_coef_, dropout_keep_prob_;
    private long seed_;
    private String name_;
    private float train_percentage_;

    protected LinkedList<TrainParamsModifier> modifiable_params = new LinkedList<>();

    protected static HashMap<String, TrainParamsModifier> params;

    static {
        params = new HashMap<>();
        params.put("epochs_num", new TrainParamsModifier() {
            @Override
            public IBuilder<TrainParams> modify(IBuilder<TrainParams> p, Object value) {
                return ((Builder) p).epochs_num((Integer) value);
            }

            @Override
            public Object parse(String s) {
                return Integer.parseInt(s);
            }

            @Override
            public void set(TrainParams p, Object value) {
                p.epochs_num_ = (int) value;
            }
        });
        params.put("batch_size", new TrainParamsModifier() {
            @Override
            public IBuilder<TrainParams> modify(IBuilder<TrainParams> p, Object value) {
                return ((Builder) p).batch_size((Integer) value);
            }

            @Override
            public Object parse(String s) {
                return Integer.parseInt(s);
            }

            @Override
            public void set(TrainParams p, Object value) {
                p.batch_size_ = (int) value;
            }
        });
        params.put("normalize_features", new TrainParamsModifier() {
            @Override
            public IBuilder<TrainParams> modify(IBuilder<TrainParams> p, Object value) {
                return ((Builder) p).normalize_features((Boolean) value);
            }

            @Override
            public Object parse(String s) {
                return Boolean.parseBoolean(s);
            }

            @Override
            public void set(TrainParams p, Object value) {
                p.normalize_features_ = (boolean) value;
            }
        });
        params.put("shuffle_batches", new TrainParamsModifier() {
            @Override
            public IBuilder<TrainParams> modify(IBuilder<TrainParams> p, Object value) {
                return ((Builder) p).shuffle_batches((Boolean) value);
            }

            @Override
            public Object parse(String s) {
                return Boolean.parseBoolean(s);
            }

            @Override
            public void set(TrainParams p, Object value) {
                p.shuffle_batches_ = (boolean) value;
            }
        });
        params.put("batch_norm", new TrainParamsModifier() {
            @Override
            public IBuilder<TrainParams> modify(IBuilder<TrainParams> p, Object value) {
                return ((Builder) p).batch_norm((Boolean) value);
            }

            @Override
            public Object parse(String s) {
                return Boolean.parseBoolean(s);
            }

            @Override
            public void set(TrainParams p, Object value) {
                p.batch_norm_ = (boolean) value;
            }
        });
        params.put("learning_rate", new TrainParamsModifier() {
            @Override
            public IBuilder<TrainParams> modify(IBuilder<TrainParams> p, Object value) {
                return ((Builder) p).learning_rate((Double) value);
            }

            @Override
            public Object parse(String s) {
                return Double.parseDouble(s);
            }

            @Override
            public void set(TrainParams p, Object value) {
                p.learning_rate_ = (double) value;
            }
        });
        params.put("decay_rate", new TrainParamsModifier() {
            @Override
            public IBuilder<TrainParams> modify(IBuilder<TrainParams> p, Object value) {
                return ((Builder) p).decay_rate((Double) value);
            }

            @Override
            public Object parse(String s) {
                return Double.parseDouble(s);
            }

            @Override
            public void set(TrainParams p, Object value) {
                p.decay_rate_ = (double) value;
            }
        });
        params.put("decay_step", new TrainParamsModifier() {
            @Override
            public IBuilder<TrainParams> modify(IBuilder<TrainParams> p, Object value) {
                return ((Builder) p).decay_step((Integer) value);
            }

            @Override
            public Object parse(String s) {
                return Integer.parseInt(s);
            }

            @Override
            public void set(TrainParams p, Object value) {
                p.decay_step_ = (int) value;
            }
        });
        params.put("regularization_coef", new TrainParamsModifier() {
            @Override
            public IBuilder<TrainParams> modify(IBuilder<TrainParams> p, Object value) {
                return ((Builder) p).regularization_coef((Double) value);
            }

            @Override
            public Object parse(String s) {
                return Double.parseDouble(s);
            }

            @Override
            public void set(TrainParams p, Object value) {
                p.regularization_coef_ = (double) value;
            }
        });
        params.put("dropout_keep_prob", new TrainParamsModifier() {
            @Override
            public IBuilder<TrainParams> modify(IBuilder<TrainParams> p, Object value) {
                return ((Builder) p).dropout_keep_prob((Double) value);
            }

            @Override
            public Object parse(String s) {
                return Double.parseDouble(s);
            }

            @Override
            public void set(TrainParams p, Object value) {
                p.dropout_keep_prob_ = (double) value;
            }
        });
        params.put("seed", new TrainParamsModifier() {
            @Override
            public IBuilder<TrainParams> modify(IBuilder<TrainParams> p, Object value) {
                return ((Builder) p).seed((Long) value);
            }

            @Override
            public Object parse(String s) {
                return Long.parseLong(s);
            }

            @Override
            public void set(TrainParams p, Object value) {
                p.seed_ = (long) value;
            }
        });
        params.put("train_percentage", new TrainParamsModifier() {
            @Override
            public IBuilder<TrainParams> modify(IBuilder<TrainParams> p, Object value) {
                return ((Builder) p).train_percentage((Float) value);
            }

            @Override
            public Object parse(String s) {
                return Float.parseFloat(s);
            }

            @Override
            public void set(TrainParams p, Object value) {
                p.train_percentage_ = (float) value;
            }
        });
    }

    /**
     * Create an empty params object. Used when parsing from a string.
     */
    public TrainParams() {
    }

    public TrainParams(int input_size, int output_size, int epochs_num, int batch_size, boolean normalize_features, boolean shuffle_batches, boolean batch_norm, double learning_rate, double decay_rate, int decay_step, double regularization_coef, double dropout_keep_prob, long seed, String name, float train_percentage) {
        input_size_ = input_size;
        output_size_ = output_size;
        epochs_num_ = epochs_num;
        batch_size_ = batch_size;
        normalize_features_ = normalize_features;
        shuffle_batches_ = shuffle_batches;
        batch_norm_ = batch_norm;
        learning_rate_ = learning_rate;
        decay_rate_ = decay_rate;
        decay_step_ = decay_step;
        regularization_coef_ = regularization_coef;
        dropout_keep_prob_ = dropout_keep_prob;
        seed_ = seed;
        name_ = name;
        train_percentage_ = train_percentage;
    }

    public TrainParams(TrainParams p) {
        input_size_ = p.input_size_;
        output_size_ = p.output_size_;
        epochs_num_ = p.epochs_num_;
        batch_size_ = p.batch_size_;
        normalize_features_ = p.normalize_features_;
        shuffle_batches_ = p.shuffle_batches_;
        batch_norm_ = p.batch_norm_;
        learning_rate_ = p.learning_rate_;
        decay_rate_ = p.decay_rate_;
        decay_step_ = p.decay_step_;
        regularization_coef_ = p.regularization_coef_;
        dropout_keep_prob_ = p.dropout_keep_prob_;
        seed_ = p.seed_;
        name_ = p.name_;
        train_percentage_ = p.train_percentage_;
    }

    public int input_size() {
        return input_size_;
    }

    public void input_size(int size) {
        input_size_ = size;
    }

    public int output_size() {
        return output_size_;
    }

    public void output_size(int size) {
        output_size_ = size;
    }

    public int epochs_num() {
        return epochs_num_;
    }

    public int batch_size() {
        return batch_size_;
    }

    public boolean normalize_features() {
        return normalize_features_;
    }

    public boolean shuffle_batches() {
        return shuffle_batches_;
    }

    public boolean batch_norm() {
        return batch_norm_;
    }

    public double learning_rate() {
        return learning_rate_;
    }

    public double decay_rate() {
        return decay_rate_;
    }

    public double decay_step() {
        return decay_step_;
    }

    public double regularization_coef() {
        return regularization_coef_;
    }

    public double dropout_keep_prob() {
        return dropout_keep_prob_;
    }

    public long seed() {
        return seed_;
    }

    public void seed(long seed) {
        seed_ = seed;
    }

    public String name() {
        return name_;
    }

    public void name(String name) {
        name_ = name;
    }

    public float train_percentage() {
        return train_percentage_;
    }

    /**
     * Parses entire text line-by-line and populates <code>this</code> with parameters.
     */
    @Override
    public boolean parse(String line) {
        line = line.trim();
        if (line.charAt(0) == '#') // Skip comments.
            return true;

        // Parse key-value pair.
        Matcher m = Utilities.KEY_VALUE_REGEX.matcher(line);
        if (!m.find()) // Return false if not parsable.
            return false;

        String key = m.group(1).trim();
        String value = m.group(2).trim();

        TrainParamsModifier mod = params.get(key);
        if (mod == null) // Return false if unknown parameter.
            return false;

        // Try parsing array.
        m = Utilities.ARRAY_REGEX.matcher(value);
        if (m.find()) { // Parse values array.
            String[] parts = m.group(1).split(Utilities.ARRAY_SEPARATOR);
            LinkedList<Object> list = new LinkedList<>();
            for (String s : parts) {
                list.add(mod.parse(s));
            }
            mod.setValues(list);
            modifiable_params.add(mod);
        } else { // Parse single value.
            mod.set(this, mod.parse(value));
        }
        return true;
    }

    @Override
    public String serialize() {
        return new StringBuilder()
                .append("input_size").append('\t').append(input_size_).append("  # Generated").append('\n')
                .append("output_size").append('\t').append(output_size_).append("  # Generated").append('\n')
                .append("epochs_num").append('\t').append(epochs_num_).append('\n')
                .append("batch_size").append('\t').append(batch_size_).append('\n')
                .append("normalize_features").append('\t').append(normalize_features_).append('\n')
                .append("shuffle_batches").append('\t').append(shuffle_batches_).append('\n')
                .append("batch_norm").append('\t').append(batch_norm_).append('\n')
                .append("learning_rate").append('\t').append(learning_rate_).append('\n')
                .append("decay_rate").append('\t').append(decay_rate_).append('\n')
                .append("decay_step").append('\t').append(decay_step_).append('\n')
                .append("regularization_coef").append('\t').append(regularization_coef_).append('\n')
                .append("dropout_keep_prob").append('\t').append(dropout_keep_prob_).append('\n')
                .append("seed").append('\t').append(seed_).append('\n')
                .append("name").append('\t').append(name_).append("  # Generated").append('\n')
                .append("train_percentage").append('\t').append(train_percentage_).append('\n')
                .toString();
    }

    @Override
    public String toString() {
        return serialize();
    }

    public GridSearch.IModifier<TrainParams>[] getModifiers() {
        return modifiable_params.toArray(new TrainParamsModifier[]{});
    }

    protected static abstract class TrainParamsModifier implements GridSearch.IModifier<TrainParams> {
        private LinkedList<Object> values;

        public TrainParamsModifier() {
        }

        public void setValues(LinkedList<Object> values) {
            this.values = values;
        }

        @Override
        public Object[] getValues() {
            return values.toArray();
        }

        public abstract Object parse(String s);

        public abstract void set(TrainParams p, Object value);
    }

    public static class Builder implements IBuilder<TrainParams> {
        private int input_size_ = -1, output_size_ = -1;
        private int epochs_num_ = -1, batch_size_ = 1;
        private boolean shuffle_batches_ = false, normalize_features_ = false, batch_norm_ = false;
        private double learning_rate_ = 1, decay_rate_ = 1;
        private int decay_step_ = 1;
        private double regularization_coef_ = 0, dropout_keep_prob_ = 1;
        private long seed_ = 42;
        private String name_ = "Name";
        private float train_percentage_;

        public TrainParams build() {
//            if (input_size_ < 0 || output_size_ < 0) {
//                throw new IllegalArgumentException("Input and output sizes must be defined!");
//            }
//            if (epochs_num_ < 0) {
//                throw new IllegalArgumentException("Epochs number must be defined!");
//            }

            return new TrainParams(input_size_, output_size_, epochs_num_, batch_size_,
                    normalize_features_, shuffle_batches_, batch_norm_,
                    learning_rate_, decay_rate_, decay_step_,
                    regularization_coef_, dropout_keep_prob_,
                    seed_, name_, train_percentage_);
        }

        public Builder input_size(int size) {
            input_size_ = size;
            return this;
        }

        public Builder output_size(int size) {
            output_size_ = size;
            return this;
        }

        public Builder epochs_num(int number) {
            epochs_num_ = number;
            return this;
        }

        public Builder batch_size(int size) {
            batch_size_ = size;
            return this;
        }

        public Builder normalize_features(boolean normalize) {
            normalize_features_ = normalize;
            return this;
        }

        public Builder shuffle_batches(boolean shuffle) {
            shuffle_batches_ = shuffle;
            return this;
        }

        public Builder batch_norm(boolean batch_norm) {
            batch_norm_ = batch_norm;
            return this;
        }

        public Builder learning_rate(double value) {
            learning_rate_ = value;
            return this;
        }

        public Builder decay_rate(double value) {
            decay_rate_ = value;
            return this;
        }

        public Builder decay_step(int value) {
            decay_step_ = value;
            return this;
        }

        public Builder regularization_coef(double value) {
            regularization_coef_ = value;
            return this;
        }

        public Builder dropout_keep_prob(double value) {
            dropout_keep_prob_ = value;
            return this;
        }

        public Builder train_percentage(float train_percentage) {
            train_percentage_ = train_percentage;
            return this;
        }

        public Builder seed(long value) {
            seed_ = value;
            return this;
        }

        public long seed() {
            return seed_;
        }

        public Builder name(String name) {
            name_ = name;
            return this;
        }

        public Builder cloneFrom(@NotNull TrainParams p) {
            input_size_ = p.input_size_;
            output_size_ = p.output_size_;
            epochs_num_ = p.epochs_num_;
            batch_size_ = p.batch_size_;
            normalize_features_ = p.normalize_features_;
            shuffle_batches_ = p.shuffle_batches_;
            batch_norm_ = p.batch_norm_;
            learning_rate_ = p.learning_rate_;
            decay_rate_ = p.decay_rate_;
            decay_step_ = p.decay_step_;
            regularization_coef_ = p.regularization_coef_;
            dropout_keep_prob_ = p.dropout_keep_prob_;
            seed_ = p.seed_;
            name_ = p.name_;
            return this;
        }
    }
}
