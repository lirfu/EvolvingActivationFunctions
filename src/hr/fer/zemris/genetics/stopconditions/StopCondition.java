package hr.fer.zemris.genetics.stopconditions;

import hr.fer.zemris.genetics.Result;
import hr.fer.zemris.utils.ISerializable;
import hr.fer.zemris.utils.Utilities;

public class StopCondition implements ISerializable {
    private Long max_iterations_;
    private Long max_evaluations_;
    private Long max_time_;
    private Double min_fitness_;
    private Double max_fitness_;

    public StopCondition() {
    }

    public StopCondition(Long max_iter_cond, Long max_eval_cond, Long max_time_cond, Double min_fitness_cond, Double max_fitness_cond) {
        max_iterations_ = max_iter_cond;
        max_evaluations_ = max_eval_cond;
        max_time_ = max_time_cond;
        min_fitness_ = min_fitness_cond;
        max_fitness_ = max_fitness_cond;
    }

    public boolean isSatisfied(Result r) {
        return (max_iterations_ != null && max_iterations_ <= r.getIterations()) ||
                (min_fitness_ != null && min_fitness_ >= r.getBest().getFitness()) ||
                (max_fitness_ != null && max_fitness_ <= r.getBest().getFitness()) ||
                (max_evaluations_ != null && max_evaluations_ <= r.getEvaluations()) ||
                (max_time_ != null && max_time_ <= r.getElapsedTime());
    }

    public String report(Result r) {
        String s = "";
        if (max_iterations_ != null && max_iterations_ <= r.getIterations()) {
            s += "Max iterations achieved!\n";
        }
        if (min_fitness_ != null && min_fitness_ >= r.getBest().getFitness()) {
            s += "Min fitness achieved!\n";
        }
        if (max_fitness_ != null && max_fitness_ <= r.getBest().getFitness()) {
            s += "Max fitness achieved!\n";
        }
        if (max_evaluations_ != null && max_evaluations_ <= r.getEvaluations()) {
            s += "Max evaluations achieved!\n";
        }
        if (max_time_ != null && max_time_ <= r.getElapsedTime()) {
            s += "Max time achieved!\n";
        }
        if (s.isEmpty()) return "No condition satisfied!";
        return s;
    }

    @Override
    public boolean parse(String line) {
        String[] parts = line.split(Utilities.PARSER_REGEX);
        switch (parts[0]) {
            case "max_iterations":
                max_iterations_ = Long.parseLong(parts[1]);
                return true;
            case "max_evaluations":
                max_evaluations_ = Long.parseLong(parts[1]);
                return true;
            case "max_time":
                max_time_ = Long.parseLong(parts[1]);
                return true;
            case "min_fitness":
                min_fitness_ = Double.parseDouble(parts[1]);
                return true;
            case "max_fitness":
                max_fitness_ = Double.parseDouble(parts[1]);
                return true;
        }
        return false;
    }

    @Override
    public String serialize() {
        StringBuilder sb = new StringBuilder();
        if (max_iterations_ != null)
            sb.append("max_iterations").append('\t').append(max_iterations_).append('\n');
        if (max_evaluations_ != null)
            sb.append("max_evaluations").append('\t').append(max_evaluations_).append('\n');
        if (max_time_ != null)
            sb.append("max_time").append('\t').append(max_time_).append('\n');
        if (min_fitness_ != null)
            sb.append("min_fitness").append('\t').append(min_fitness_).append('\n');
        if (max_fitness_ != null)
            sb.append("max_fitness").append('\t').append(max_fitness_).append('\n');
        return sb.toString();
    }

    public static class Builder {
        private Long max_iterations_;
        private Long max_evals_condition_;
        private Long max_time_condition_;
        private Double min_fitness_;
        private Double max_fitness_;

        public StopCondition build() {
            return new StopCondition(max_iterations_, max_evals_condition_, max_time_condition_, min_fitness_, max_fitness_);
        }

        public Builder setMaxIterations(long max_iterations) {
            max_iterations_ = max_iterations;
            return this;
        }

        public Builder setMaxEvaluations(long max_evaluations) {
            max_evals_condition_ = max_evaluations;
            return this;
        }

        public Builder setMaxTimeMs(long max_time) {
            max_time_condition_ = max_time;
            return this;
        }

        public Builder setMinFitness(double min_fitness) {
            min_fitness_ = min_fitness;
            return this;
        }

        public Builder setMaxFitness(double max_fitness) {
            max_fitness_ = max_fitness;
            return this;
        }
    }
}
