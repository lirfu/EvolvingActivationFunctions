package hr.fer.zemris.genetics.stopconditions;

import hr.fer.zemris.genetics.Result;

public class StopCondition {
    private final Long max_iterations_;
    private final Long max_evals_condition_;
    private final Long max_time_condition_;
    private final Double min_fitness_;
    private final Double max_fitness_;

    public StopCondition(Long max_iter_cond, Long max_eval_cond, Long max_time_cond, Double min_fitness_cond, Double max_fitness_cond) {
        max_iterations_ = max_iter_cond;
        max_evals_condition_ = max_eval_cond;
        max_time_condition_ = max_time_cond;
        min_fitness_ = min_fitness_cond;
        max_fitness_ = max_fitness_cond;
    }

    public boolean isSatisfied(Result r) {
        return (max_iterations_ != null && max_iterations_ <= r.getIterations()) ||
                (min_fitness_ != null && min_fitness_ >= r.getBest().getFitness()) ||
                (max_fitness_ != null && max_fitness_ <= r.getBest().getFitness()) ||
                (max_evals_condition_ != null && max_evals_condition_ <= r.getEvaluations()) ||
                (max_time_condition_ != null && max_time_condition_ <= r.getElapsedTime());
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
        if (max_evals_condition_ != null && max_evals_condition_ <= r.getEvaluations()) {
            s += "Max evaluations achieved!\n";
        }
        if (max_time_condition_ != null && max_time_condition_ <= r.getElapsedTime()) {
            s += "Max time achieved!\n";
        }
        if (s.isEmpty()) return "No condition satisfied!";
        return s;
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
