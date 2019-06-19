package hr.fer.zemris.utils;

public class Triple<T, V, S> {
    private T first_;
    private V second_;
    private S third_;

    public Triple(T key, V val, S extra) {
        this.first_ = key;
        this.second_ = val;
        this.third_ = extra;
    }

    public Triple(Triple<T, V, S> p) {
        first_ = p.first_;
        second_ = p.second_;
        third_ = p.third_;
    }

    public T getFirst() {
        return first_;
    }

    public V getSecond() {
        return second_;
    }

    public S getThird() {
        return third_;
    }

    @Override
    public String toString() {
        return first_.toString() + '\t' + second_.toString() + '\t' + third_.toString();
    }
}
