package hr.fer.zemris.neurology.dl4j;

import hr.fer.zemris.data.ISerializable;
import org.jetbrains.annotations.NotNull;

/**
 * Immutable wrapper for train parameters.
 */
public class TrainParams implements ISerializable {
    private int input_size_, output_size_;
    private int epochs_num_, batch_size_;
    private boolean normalize_features_, shuffle_batches_;
    private double learning_rate_, decay_rate_;
    private int decay_step_;
    private double regularization_coef_, dropout_keep_prob_;
    private long seed_;
    private String name_;

    /**
     * Create an empty params object. Used when parsing from a string.
     */
    public TrainParams() {
    }

    public TrainParams(int input_size, int output_size, int epochs_num, int batch_size, boolean normalize_features, boolean shuffle_batches, double learning_rate, double decay_rate, int decay_step, double regularization_coef, double dropout_keep_prob, long seed, String name) {
        input_size_ = input_size;
        output_size_ = output_size;
        epochs_num_ = epochs_num;
        batch_size_ = batch_size;
        normalize_features_ = normalize_features;
        shuffle_batches_ = shuffle_batches;
        learning_rate_ = learning_rate;
        decay_rate_ = decay_rate;
        decay_step_ = decay_step;
        regularization_coef_ = regularization_coef;
        dropout_keep_prob_ = dropout_keep_prob;
        seed_ = seed;
        name_ = name;
    }


    public int input_size() {
        return input_size_;
    }

    public int output_size() {
        return output_size_;
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

    public String name() {
        return name_;
    }

    @Override
    public void parse(String s) {
        for (String line : s.split("\n")) {
            String[] parts = line.split("\t");
            switch (parts[0]) {
                case "input_size":
                    input_size_ = Integer.parseInt(parts[1]);
                    break;
                case "output_size":
                    output_size_ = Integer.parseInt(parts[1]);
                    break;
                case "epochs_num":
                    epochs_num_ = Integer.parseInt(parts[1]);
                    break;
                case "batch_size":
                    batch_size_ = Integer.parseInt(parts[1]);
                    break;
                case "normalize_features":
                    normalize_features_ = Boolean.parseBoolean(parts[1]);
                    break;
                case "shuffle_batches":
                    shuffle_batches_ = Boolean.parseBoolean(parts[1]);
                    break;
                case "learning_rate":
                    learning_rate_ = Double.parseDouble(parts[1]);
                    break;
                case "decay_rate":
                    decay_rate_ = Double.parseDouble(parts[1]);
                    break;
                case "decay_step":
                    decay_step_ = Integer.parseInt(parts[1]);
                    break;
                case "regularization_coef":
                    regularization_coef_ = Double.parseDouble(parts[1]);
                    break;
                case "dropout_keep_prob":
                    dropout_keep_prob_ = Double.parseDouble(parts[1]);
                    break;
                case "seed":
                    seed_ = Long.parseLong(parts[1]);
                    break;
                case "name":
                    name_ = parts[1];
                    break;
            }
        }
    }

    @Override
    public String serialize() {
        return new StringBuilder()
                .append("input_size").append('\t').append(input_size_).append('\n')
                .append("output_size").append('\t').append(output_size_).append('\n')
                .append("epochs_num").append('\t').append(epochs_num_).append('\n')
                .append("batch_size").append('\t').append(batch_size_).append('\n')
                .append("normalize_features").append('\t').append(normalize_features_).append('\n')
                .append("shuffle_batches").append('\t').append(shuffle_batches_).append('\n')
                .append("learning_rate").append('\t').append(learning_rate_).append('\n')
                .append("decay_rate").append('\t').append(decay_rate_).append('\n')
                .append("decay_step").append('\t').append(decay_step_).append('\n')
                .append("regularization_coef").append('\t').append(regularization_coef_).append('\n')
                .append("dropout_keep_prob").append('\t').append(dropout_keep_prob_).append('\n')
                .append("seed").append('\t').append(seed_).append('\n')
                .append("name").append('\t').append(name_).append('\n')
                .toString();
    }

    @Override
    public String toString() {
        return serialize();
    }

    public static class Builder {
        private int input_size_ = -1, output_size_ = -1;
        private int epochs_num_ = -1, batch_size_ = 1;
        private boolean shuffle_batches_ = false, normalize_features_ = false;
        private double learning_rate_ = 0.1, decay_rate_ = 1;
        private int decay_step_ = 1;
        private double regularization_coef_ = 0, dropout_keep_prob_ = 1;
        private long seed_ = 42;
        private String name_ = "Model";

        public TrainParams build() {
            if (input_size_ < 0 || output_size_ < 0) {
                throw new IllegalArgumentException("Input and output sizes must be defined!");
            }
            if (epochs_num_ < 0) {
                throw new IllegalArgumentException("Epochs number must be defined!");
            }

            return new TrainParams(input_size_, output_size_, epochs_num_, batch_size_, normalize_features_, shuffle_batches_, learning_rate_, decay_rate_, decay_step_, regularization_coef_, dropout_keep_prob_, seed_, name_);
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

        public Builder seed(long value) {
            seed_ = value;
            return this;
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
