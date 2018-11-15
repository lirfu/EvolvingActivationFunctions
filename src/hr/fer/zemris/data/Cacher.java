package hr.fer.zemris.data;

import com.sun.istack.internal.NotNull;
import hr.fer.zemris.data.modifiers.IModifier;
import hr.fer.zemris.data.primitives.DataPair;
import hr.fer.zemris.utils.Pair;

import java.util.ArrayList;

/**
 * Caches the whole input stream into RAM.
 * Applies modifiers to the data such as normalization, randomization, etc.
 */
public class Cacher extends APipe<DataPair, DataPair> {
    private IModifier[] data_modifiers_;
    private DataPair[] data_;
    private int index = 0;

    /**
     * Caches the input stream into RAM.
     */
    public Cacher(@NotNull APipe<?, DataPair> parent) {
        this(parent, new IModifier[]{});
    }

    /**
     * Caches the input stream to the specified file.
     * Applies modifiers to the dataset in the order they are specified.
     */
    public Cacher(@NotNull APipe<?, DataPair> parent, @NotNull IModifier[] data_modifiers) {
        parent_ = parent;
        data_modifiers_ = data_modifiers;

        // Load the data into memory.
        ArrayList<DataPair> data = new ArrayList<>();
        DataPair d;
        while ((d = parent.get()) != null)
            data.add(d);
        data_ = new DataPair[]{};
        data_ = data.toArray(data_);

        // Apply modifiers to data.
        for (IModifier m : data_modifiers) {
            m.apply(data_);
        }
    }

    /**
     * Instead of generating the data all over again, just use the original reference.
     * This saves memory when cloning (shares data) and since modifiers aren't re-applied (important for maintaining randomization).
     */
    private Cacher(@NotNull APipe<?, DataPair> parent, @NotNull IModifier[] data_modifiers, DataPair[] data) {
        parent_ = parent;
        data_modifiers_ = data_modifiers;
        data_ = data;
    }


    /**
     * Returns cached data.
     */
    @Override
    public DataPair get() {
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
     * Apply the given modifier on the dataset.
     *
     * @param modifier Modifier applied to the dataset.
     */
    public void applyModifier(@NotNull IModifier modifier) {
        modifier.apply(data_);
    }

    /**
     * Sends the reset signal to parent and resets the internal index.
     */
    public void hardReset() {
        index = 0;
        parent_.reset();
    }

    /**
     * Sets parent to null, effectively releasing this resource.
     * Useful for releasing memory in concurrent applications.
     */
    public void releaseParent() {
        parent_ = null;
    }

    /**
     * Returns a clone of this pipe.
     * The internal data is shared with clones (for memory efficiency) and the modifiers are not re-applied.
     */
    @Override
    public Cacher clone() {
        return new Cacher(parent_, data_modifiers_, data_);
    }
}
