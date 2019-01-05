package hr.fer.zemris.genetics;

import hr.fer.zemris.utils.ISerializable;

public abstract class Operator<T extends Operator> implements ISerializable {
    protected static final String SPLIT_REGEX = "\t ,:";
    private int importance = 1;

    protected Operator() {
    }

    public abstract String getName();

    public int getImportance() {
        return importance;
    }

    /**
     * Sets importance of operator. Higher importance means higher chance of using it instead of others (if specified). Level must be within [0, 10].
     */
    public T setImportance(int level) {
        if (level < 0 || level > 10)
            throw new IllegalArgumentException("Level must be within [0, 10]!");
        importance = level;
        return (T) this;
    }

    @Override
    public void parse(String line) {
        String[] p = line.split(SPLIT_REGEX);
        if (p[0].equals(getName())) {
            importance = Integer.parseInt(p[1]);
        }
    }

    @Override
    public String serialize() {
        return serializeKeyVal(getName(), String.valueOf(importance));
    }

    protected String serializeKeyVal(String key, String val) {
        return key + '\t' + val + '\n';
    }
}
