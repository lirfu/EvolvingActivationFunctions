package hr.fer.zemris.data;

import hr.fer.zemris.data.primitives.BatchPair;
import hr.fer.zemris.data.primitives.DataPair;
import hr.fer.zemris.utils.Pair;

/**
 * Constructs batches from the input stream.
 */
public class Batcher extends APipe<DataPair, BatchPair> {
    private int batch_size_;

    /**
     * Reads the data stream from parent and constructs batches of specified size.
     *
     * @param parent     Data stream source.
     * @param batch_size Size of the constructed batches. Last batch might be smaller then the specified size
     *                   (depending on the stream size).
     */
    public Batcher(APipe<?, DataPair> parent, int batch_size) {
        parent_ = parent;
        batch_size_ = batch_size;
    }

    /**
     * Reads the parent stream until the batch is constructed (has <code>batch_size</code> datapoints).
     * Last batch might be smaller then the specified size (depending on the stream size).
     * After the last batch was constructed, this method returns <code>null</code>.
     */
    @Override
    public BatchPair next() {
        int this_batch_size = 0, data_dim = 0;
        float[][] inputs = null;
        float[] labels = new float[batch_size_];

        for (int i = 0; i < batch_size_; i++) {
            Pair<float[], Float> pair = parent_.next();
            if (pair == null) break; // End of stream.
            this_batch_size++;

            if (inputs == null) { // Initialize array size.
                data_dim = pair.getKey().length;
                inputs = new float[batch_size_][data_dim];
            }

            inputs[i] = pair.getKey();
            labels[i] = pair.getVal();
        }

        if (inputs == null) return null; // End of stream.

        if (this_batch_size < batch_size_) { // Trim if batch is not filled.
            float[][] short_inputs = new float[this_batch_size][data_dim];
            float[] short_labels = new float[this_batch_size];
            System.arraycopy(inputs, 0, short_inputs, 0, this_batch_size);
            System.arraycopy(labels, 0, short_labels, 0, this_batch_size);
            return new BatchPair(short_inputs, short_labels);
        }
        return new BatchPair(inputs, labels);
    }

    /**
     * Resets the parent.
     */
    @Override
    public void reset() {
        parent_.reset();
    }

    @Override
    public Batcher clone() {
        return new Batcher(parent_, batch_size_);
    }
}
