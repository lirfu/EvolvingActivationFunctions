package hr.fer.zemris.genetics;


public interface Initializer<T extends Genotype> {
    public void initialize(T genotype);
}
