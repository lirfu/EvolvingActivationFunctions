package hr.fer.zemris.data;

import org.jetbrains.annotations.NotNull;

import java.util.ArrayList;
import java.util.Iterator;

public class Cacher<T> extends APipe<T, T> implements Iterable<T> {
    private int index_ = 0;
    private ArrayList<T> data_;

    public Cacher(APipe<?, T> parent) {
        parent_ = parent;
        data_ = new ArrayList<>();
        T t;
        while ((t = parent.next()) != null) {
            data_.add(t);
        }
    }

    private Cacher(Cacher<T> c) {
        data_ = c.data_;
    }

    public boolean hasNext() {
        return index_ < data_.size();
    }

    @Override
    public T next() {
        if (index_ >= data_.size()) {
            return null;
        }
        return data_.get(index_++);
    }

    @Override
    public void reset() {
        index_ = 0;
        if (parent_ != null) {
            parent_.reset();
        }
    }

    public void releaseParent() {
        parent_ = null;
    }

    @Override
    public APipe<T, T> clone() {
        return new Cacher<>(this);
    }

    @NotNull
    @Override
    public Iterator<T> iterator() {
        return data_.iterator();
    }
}
