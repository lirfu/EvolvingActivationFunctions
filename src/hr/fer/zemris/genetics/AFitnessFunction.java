package hr.fer.zemris.genetics;

public abstract class AFitnessFunction {
    private long evaluations = 0;
    public void resetEvals(){
        evaluations=0;
    }

    public long getEvaluations() {
        return evaluations;
    }

    public final double evaluate(Genotype g){
        evaluations++;
        return performEvaluate(g);
    }

    public abstract double performEvaluate(Genotype g);
}
