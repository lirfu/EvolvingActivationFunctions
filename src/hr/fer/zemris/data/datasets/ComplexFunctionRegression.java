package hr.fer.zemris.data.datasets;

import hr.fer.zemris.data.ADataGenerator;
import hr.fer.zemris.data.APipe;
import hr.fer.zemris.data.DatasetDescriptor;
import hr.fer.zemris.data.IDescriptableDS;
import hr.fer.zemris.data.primitives.DataPair;

public class ComplexFunctionRegression extends ADataGenerator {
    private DataPair[] data_;
    private int index_;
    private DatasetDescriptor descriptor_;

    public ComplexFunctionRegression(int size) {
        descriptor_ = new DatasetDescriptor("ComplexFunctionRegression", 3, 1, size);
        data_ = new DataPair[size];
        for (int i = 0; i < size; i++) {
            float xx = (float) (i * 2 * Math.PI / size);
            data_[i] = new DataPair(new float[]{xx, i, 5 * xx * i}, new float[]{(float) (0.2f * Math.sin(xx) + 0.2f * Math.sin(2 * xx + Math.PI / 7) + 0.5f)});
        }
    }

    @Override
    public DataPair next() {
        if (index_ >= data_.length) return null;
        return data_[index_++];
    }

    @Override
    public void reset() {
        index_ = 0;
    }

    @Override
    public APipe<Object, DataPair> clone() {
        return new ComplexFunctionRegression(data_.length);
    }

    @Override
    public DatasetDescriptor describe() {
        return descriptor_;
    }
}