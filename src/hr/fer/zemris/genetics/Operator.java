package hr.fer.zemris.genetics;

public abstract class Operator {
    private int importance = 1;

    protected Operator() {
    }

    public int getImportance() {
        return importance;
    }

    /**
     * Sets importance of operator. Higher importance means higher chance of using it instead of others (if specified). Level must be within [0, 10].
     */
    public Operator setImportance(int level) {
        if (level < 0 || level > 10)
            throw new IllegalArgumentException("Level must be within [0, 10]!");
        importance = level;
        return this;
    }
}
