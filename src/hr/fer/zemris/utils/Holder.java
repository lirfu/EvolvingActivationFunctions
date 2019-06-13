package hr.fer.zemris.utils;

/**
 * Utility class for holding an instance while defined as final. Usages include holding an instance in multithreaded setup.
 */
public class Holder<T> {
    private T item_;

    public Holder() {
    }

    public Holder(T item) {
        item_ = item;
    }

    public synchronized void set(T item) {
        item_ = item;
    }

    public T get() {
        return item_;
    }

    public boolean isDefined() {
        return item_ != null;
    }
}
