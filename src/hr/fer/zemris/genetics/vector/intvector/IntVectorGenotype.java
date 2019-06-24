package hr.fer.zemris.genetics.vector.intvector;

import hr.fer.zemris.genetics.Genotype;
import hr.fer.zemris.genetics.vector.AVectorGenotype;

import java.util.Random;

public class IntVectorGenotype extends AVectorGenotype<Integer> {
    private int max_val_, min_val_;
    private Integer[] indices_;

    public IntVectorGenotype(int size, int min_val, int max_val) {
        indices_ = new Integer[size];
        min_val_ = min_val;
        max_val_ = max_val;
    }

    public IntVectorGenotype(IntVectorGenotype g) {
        super(g);
        min_val_ = g.min_val_;
        max_val_ = g.max_val_;
        indices_ = new Integer[g.size()];
        System.arraycopy(g.indices_, 0, indices_, 0, indices_.length);
    }

    @Override
    public Integer get(int index) {
        return indices_[index];
    }

    @Override
    public void set(int index, Integer value) {
        indices_[index] = value;
    }

    @Override
    public int size() {
        return indices_.length;
    }

    @Override
    public Genotype copy() {
        return new IntVectorGenotype(this);
    }

    @Override
    public Integer generateParameter(Random rand) {
        return min_val_ + rand.nextInt(max_val_ - min_val_ + 1);
    }

    @Override
    public boolean parse(String line) {
        String[] parts = line.split(",");
        indices_ = new Integer[parts.length];
        for (int i = 0; i < parts.length; i++) {
            indices_[i] = Integer.parseInt(parts[i]);
        }
        return true;
    }

    @Override
    public String serialize() {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < indices_.length; i++) {
            if (i > 0)
                sb.append(',');
            sb.append(indices_[i]);
        }
        return sb.toString();
    }

    @Override
    public String toString() {
        return serialize();
    }
}
