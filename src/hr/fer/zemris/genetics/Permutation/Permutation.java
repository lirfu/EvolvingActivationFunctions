package hr.fer.zemris.genetics.Permutation;

import hr.fer.zemris.genetics.Genotype;

import java.util.Random;

public class Permutation extends Genotype<Integer> {
    private Integer[] data;

    public Permutation(int size) {
        super();
        data = new Integer[size];
    }

    protected Permutation(Permutation p) {
        super(p);
        data = new Integer[p.size()];
        System.arraycopy(p.data, 0, data, 0, p.size());
    }

    @Override
    public Integer get(int index) {
        return data[index];
    }

    @Override
    public void set(int index, Integer value) {
        data[index] = value;
    }

    @Override
    public int size() {
        return data.length;
    }

    @Override
    public Genotype copy() {
        return new Permutation(this);
    }

    @Override
    public void randomize(Random rand) {
        // Init values.
        for (int i = 0; i < data.length; i++)
            data[i] = i;

        // Shuffle
        for (int i = 0; i < data.length; i++) {
            int x = rand.nextInt(data.length), y = rand.nextInt(data.length);
            Integer t = data[x];
            data[x] = data[y];
            data[y] = t;
        }
    }

    @Override
    public String stringify() {
        StringBuilder out = new StringBuilder();
        out.append('(');

        for (int i = 0; i < data.length; i++) {
            if (i > 0) out.append(", ");
            out.append(data[i]);
        }

        out.append(')');
        return out.toString();
    }

    @Override
    public Integer generateParameter(Random rand) {
        return rand.nextInt(data.length);
    }
}
