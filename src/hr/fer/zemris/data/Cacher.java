package hr.fer.zemris.data;

import com.sun.istack.internal.NotNull;
import hr.fer.zemris.data.modifiers.IModifier;
import hr.fer.zemris.utils.Pair;

import java.util.ArrayList;

/**
 * Caches the input stream into RAM.
 * Applies modifiers to the data such as normalization, randomization, etc.
 */
public class Cacher extends APipe<Pair<float[], Float>, Pair<float[], Float>> {
    private IModifier[] data_modifiers_;
    private Pair<float[], Float>[] data_;
    private int index = 0;

    /**
     * Caches the input stream into RAM.
     */
    public Cacher(@NotNull APipe<?, Pair<float[], Float>> parent) {
        this(parent, new IModifier[]{});
    }

    /**
     * Caches the input stream to the specified file.
     * Applies modifications to the dataset in the order they are specified.
     */
    public Cacher(@NotNull APipe<?, Pair<float[], Float>> parent, @NotNull IModifier[] data_modifiers) {
        parent_ = parent;
        data_modifiers_ = data_modifiers;

        // Load the data into memory.
        ArrayList<Pair<float[], Float>> data = new ArrayList<>();
        Pair<float[], Float> d;
        while ((d = parent.get()) != null)
            data.add(d);

        data_ = new Pair[]{};
        data_ = data.toArray(data_);
        // Apply modifiers to data.
        for (IModifier m : data_modifiers) {
            m.apply(data_);
        }
    }


    /**
     * Returns cached data.
     */
    @Override
    public Pair<float[], Float> get() {
        if (index >= data_.length) return null;
        return data_[index++];
    }

    /**
     * Resets the internal index to enable reusing this object.
     */
    @Override
    public void reset() {
        index = 0;
    }

    /**
     * Sends the reset signal to parent and resets the internal index.
     */
    public void hard_reset() {
        index = 0;
        parent_.reset();
    }
}
