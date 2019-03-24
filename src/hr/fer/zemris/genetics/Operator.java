package hr.fer.zemris.genetics;

import hr.fer.zemris.utils.ISerializable;
import hr.fer.zemris.utils.Utilities;

import java.util.Random;

public abstract class Operator<T extends Operator> implements ISerializable {
    private int importance_ = 1;
    protected Random r_;

    protected Operator() {
    }

    public abstract String getName();

    public int getImportance() {
        return importance_;
    }

    public T setRandom(Random random) {
        r_ = random;
        return (T) this;
    }

    /**
     * Sets importance of operator. Higher importance means higher chance of using it instead of others (if specified). Level must be within [0, 10].
     */
    public T setImportance(int level) {
        if (level < 0 || level > 10)
            throw new IllegalArgumentException("Level must be within [0, 10]!");
        importance_ = level;
        return (T) this;
    }

    @Override
    public boolean parse(String line) {
        String[] p = line.split(Utilities.KEY_VALUE_SIMPLE_REGEX);
        if (p[0].equals(getName())) {
            if (p.length > 1 && !p[1].isEmpty())
                importance_ = Integer.parseInt(p[1]);
            else
                importance_ = 1;
            return true;
        }
        return false;
    }

    @Override
    public String serialize() {
        return serializeKeyVal(getName(), String.valueOf(importance_));
    }

    protected String serializeKeyVal(String key, String val) {
        return key + '\t' + val + '\n';
    }
}
