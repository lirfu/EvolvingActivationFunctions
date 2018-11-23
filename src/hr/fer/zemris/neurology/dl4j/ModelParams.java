package hr.fer.zemris.neurology.dl4j;

public class ModelParams {
    private final int input_size_, output_size_;
    private int epochs_num_, batch_size_;
    private double learning_rate_, decay_rate_;
    private int decay_step_;
    private double regularization_coef_, dropout_keep_prob_;
    private long seed_;
    private String name_;

    public ModelParams(int input_size, int output_size, int epochs_num, int batch_size, double learning_rate, double decay_rate, int decay_step, double regularization_coef, double dropout_keep_prob, long seed, String name) {
        input_size_ = input_size;
        output_size_ = output_size;
        epochs_num_ = epochs_num;
        batch_size_ = batch_size;
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

    public static class Builder {
        private int input_size_ = -1, output_size_ = -1;
        private int epochs_num_ = -1, batch_size_ = 1;
        private double learning_rate_ = 0.1, decay_rate_ = 1;
        private int decay_step_ = 1;
        private double regularization_coef_ = 0, dropout_keep_prob_ = 1;
        private long seed_ = 42;
        private String name_ = "Model";

        public ModelParams build() {
            if (input_size_ < 0 || output_size_ < 0) {
                throw new IllegalArgumentException("Input and output sizes must be defined!");
            }
            if (epochs_num_ < 0) {
                throw new IllegalArgumentException("Epochs number must be defined!");
            }

            return new ModelParams(input_size_, output_size_, epochs_num_, batch_size_, learning_rate_, decay_rate_, decay_step_, regularization_coef_, dropout_keep_prob_, seed_, name_);
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
    }
}
