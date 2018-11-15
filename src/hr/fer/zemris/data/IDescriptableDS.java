package hr.fer.zemris.data;

public interface IDescriptableDS {
    /** Returns a descriptor of the dataset (or null if it doesn't exist). */
    public DatasetDescriptor describe();
}
