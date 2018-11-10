package hr.fer.zemris.utils;

public class Pair<T, V> {
    private T key;
    private V val;

    public Pair(T key, V val) {
        this.key = key;
        this.val = val;
    }

    public T getKey() {
        return key;
    }

    public V getVal() {
        return val;
    }
}
