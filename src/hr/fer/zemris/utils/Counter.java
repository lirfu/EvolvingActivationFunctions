package hr.fer.zemris.utils;

public class Counter {
    private int ctr_;

    public Counter() {
        ctr_ = 0;
    }

    public Counter(int value) {
        ctr_ = value;
    }

    public Counter increment() {
        ctr_++;
        return this;
    }

    public Counter decrement() {
        ctr_--;
        return this;
    }

    public int value() {
        return ctr_;
    }
}
