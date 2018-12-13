package hr.fer.zemris.utils;

public class Pair<T, V> {
    private T key_;
    private V val_;

    public Pair(T key, V val) {
        this.key_ = key;
        this.val_ = val;
    }

    public Pair(Pair<T, V> p) {
        key_ = p.key_;
        val_ = p.val_;
    }

    public T getKey() {
        return key_;
    }

    public V getVal() {
        return val_;
    }

    @Override
    public String toString() {
        return key_.toString() + '\t' + val_.toString();
    }
}
