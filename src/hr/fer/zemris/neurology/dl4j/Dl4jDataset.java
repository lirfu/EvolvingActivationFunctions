package hr.fer.zemris.neurology.dl4j;

import hr.fer.zemris.data.*;
import hr.fer.zemris.data.primitives.BatchPair;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;

import java.io.FileNotFoundException;
import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class Dl4jDataset implements DataSetIterator {
    private int input_size_, output_size_;
    private int batch_size_, instances_;
    private long seed_;
    private String name_;

    private Cacher<BatchPair> cacher_;
    private DataSetPreProcessor preprocessor_;

    public Dl4jDataset(String filepath, int batch_size, long seed) throws FileNotFoundException {
        this(new Parser(new Reader(filepath)), batch_size, seed);
    }

    public Dl4jDataset(ADataGenerator generator, int batch_size, long seed) throws FileNotFoundException {
        cacher_ = new Cacher<>(new Batcher(generator, batch_size));
        DatasetDescriptor dd = generator.describe();
        cacher_.releaseParent();

        input_size_ = dd.getAttributesNumber();
        output_size_ = dd.getClassesNumber();
        instances_ = dd.getInstancesNumber();
        batch_size_ = batch_size;
        seed_ = seed;
        name_ = dd.getName();
    }

    public void setName(String name) {
        name_ = name;
    }

    public String getName() {
        return name_;
    }

    @Override
    public int inputColumns() {
        return input_size_;
    }

    @Override
    public int totalOutcomes() {
        return output_size_;
    }

    @Override
    public int batch() {
        return batch_size_;
    }

    @Override
    public boolean hasNext() {
        return cacher_.hasNext();
    }

    @Override
    public DataSet next() {
        BatchPair b = cacher_.next();
//        System.out.println(Arrays.toString(b.getVal()[0]));
        return new DataSet(Nd4j.create(b.getKey()), Nd4j.create(b.getVal()));
    }

    @Override
    public DataSet next(int ignore) {
        return next();
    }

    @Override
    public void reset() {
        cacher_.reset();
    }

    @Override
    public boolean resetSupported() {
        return true;
    }

    @Override
    public boolean asyncSupported() {
        return false;
    }

    @Override
    public void setPreProcessor(DataSetPreProcessor dataSetPreProcessor) {
        preprocessor_ = dataSetPreProcessor;
    }

    @Override
    public DataSetPreProcessor getPreProcessor() {
        return preprocessor_;
    }

    @Override
    public List<String> getLabels() {
        ArrayList<String> list = new ArrayList<>();
        for (int i = 0; i < output_size_; i++) {
            list.add(String.valueOf(i));
        }
        return list;
    }
}
