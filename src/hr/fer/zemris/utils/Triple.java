package hr.fer.zemris.utils;

public class Triple<T, V, S> {
    private T key_;
    private V val_;
    private S extra_;

    public Triple(T key, V val, S extra) {
        this.key_ = key;
        this.val_ = val;
        this.extra_ = extra;
    }

    public Triple(Triple<T, V, S> p) {
        key_ = p.key_;
        val_ = p.val_;
        extra_ = p.extra_;
    }

    public T getKey() {
        return key_;
    }

    public V getVal() {
        return val_;
    }

    public S getExtra() {
        return extra_;
    }

    @Override
    public String toString() {
        return key_.toString() + '\t' + val_.toString() + '\t' + extra_.toString();
    }
}
