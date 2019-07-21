package hr.fer.zemris.utils;

public class Counter {
    private int ctr_;

    public Counter() {
        ctr_ = 0;
    }

    public Counter(int value) {
        ctr_ = value;
    }

    public synchronized Counter increment() {
        ctr_++;
        return this;
    }

    public synchronized Counter decrement() {
        ctr_--;
        return this;
    }

    public int value() {
        return ctr_;
    }

    @Override
    public String toString() {
        return String.valueOf(ctr_);
    }
}
