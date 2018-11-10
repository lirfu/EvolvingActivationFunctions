package hr.fer.zemris.genetics.FloatingPoint;

import hr.fer.zemris.genetics.Genotype;

import java.util.Random;

public class FloatingPoint extends Genotype<Double> {
    protected Double[] data_;
    protected double min_, max_;

    public FloatingPoint(int size) {
        super();
        data_ = new Double[size];
        min_ = -1;
        max_ = 1;
    }

    public FloatingPoint(int size, double min, double max) {
        super();
        data_ = new Double[size];
        min_ = min;
        max_ = max;
    }

    protected FloatingPoint(FloatingPoint g) {
        super(g);
        data_ = new Double[g.size()];
        System.arraycopy(g.data_, 0, this.data_, 0, g.data_.length);
        min_ = g.min_;
        max_ = g.max_;
    }

    @Override
    public Double get(int index) {
        return data_[index];
    }

    @Override
    public void set(int index, Double value) {
        data_[index] = value;
    }

    @Override
    public int size() {
        return data_.length;
    }

    public double getMin() {
        return min_;
    }

    public double getMax() {
        return max_;
    }

    @Override
    public Genotype copy() {
        return new FloatingPoint(this);
    }

    @Override
    public void randomize(Random rand) {
        for (int i = 0; i < data_.length; i++)
            data_[i] = min_ + rand.nextDouble() * (max_ - min_);
    }

    @Override
    public String stringify() {
        StringBuilder out = new StringBuilder();
        out.append('(');

        for (int i = 0; i < data_.length; i++) {
            if (i > 0) out.append(", ");
            out.append(data_[i]);
        }

        out.append(')');
        return out.toString();
    }

    @Override
    public Double generateParameter(Random rand) {
        return min_ + rand.nextDouble() * (max_ - min_);
    }
}
