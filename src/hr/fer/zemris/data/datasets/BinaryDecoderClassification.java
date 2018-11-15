package hr.fer.zemris.data.datasets;

import hr.fer.zemris.data.APipe;
import hr.fer.zemris.data.DatasetDescriptor;
import hr.fer.zemris.data.IDescriptableDS;
import hr.fer.zemris.data.primitives.DataPair;

public class BinaryDecoderClassification extends APipe<Object, DataPair> implements IDescriptableDS {
    private DataPair[] data_;
    private int index_;
    private DatasetDescriptor descriptor_ = new DatasetDescriptor("BinaryDecoderClassification", 3, 8, 8);

    public BinaryDecoderClassification() {
        data_ = new DataPair[8];
        for (int i = 0; i < 8; i++) {
            float[] x = new float[]{i / 4 % 2, i / 2 % 2, i % 2}; // Binary arrays.
            data_[i] = new DataPair(x, (float) i);
        }
    }

    @Override
    public DataPair get() {
        if (index_ >= data_.length) return null;
        return data_[index_++];
    }

    @Override
    public void reset() {
        index_ = 0;
    }

    @Override
    public APipe<Object, DataPair> clone() {
        return new BinaryDecoderClassification();
    }

    @Override
    public DatasetDescriptor describe() {
        return descriptor_;
    }
}