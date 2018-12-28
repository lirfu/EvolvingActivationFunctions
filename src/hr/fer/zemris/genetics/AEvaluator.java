package hr.fer.zemris.genetics;


public abstract class AEvaluator<T extends Genotype> {
    private long evaluations = 0;

    public void resetEvals() {
        evaluations = 0;
    }

    public long getEvaluations() {
        return evaluations;
    }

    public final double evaluate(T g) {
        evaluations++;
        return performEvaluate(g);
    }

    public abstract double performEvaluate(T g);
}
