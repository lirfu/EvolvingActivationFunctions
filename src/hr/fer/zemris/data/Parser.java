package hr.fer.zemris.data;

import com.sun.istack.internal.NotNull;
import hr.fer.zemris.utils.Pair;
import hr.fer.zemris.utils.Util;

/**
 * Reads the stream and parses it into a pair of float array input and float label (<code>Pair(float[], Float)</code>).
 * Additionally, keeps track of dataset specifics (argument, classes and instances number)
 * which can be accessed by the <code>getDatasetDescriptor()</code> method.
 */
public class Parser extends APipe<String, Pair<float[], Float>> {
    /**
     * Descriptor being constructed while reading the stream.
     * Should be used only when the whole stream was read.
     */
    private DatasetDescriptor dataset_descriptor_ = new DatasetDescriptor();
    private boolean data_started_;

    public Parser(@NotNull APipe<?, String> parent) {
        parent_ = parent;
    }

    /**
     * Reads the parent stream and constructs data pair objects from data lines.
     * Descriptor lines are processed and stored in the dataset descriptor.
     * Returns <code>null</code> if end of stream is reached.
     */
    @Override
    public Pair<float[], Float> get() {
        String s;
        while (!data_started_ && (s = parent_.get()) != null) { // Read all the descriptors until data start is reached.
            s = s.trim();
            if (s.isEmpty()) {
                continue;
            } else if (s.startsWith("@relation")) {
                dataset_descriptor_.relation = s.substring("@relation".length() + 1, s.length());

            } else if (s.startsWith("@attribute")) {

                if (s.contains("Class")) { // Classes descriptor.
                    s = s.replaceFirst(".*\\{", "");
                    s = s.substring(0, s.length() - 1);
                    dataset_descriptor_.classes_num = s.split(",").length;

                } else { // Variables descriptors.
                    dataset_descriptor_.attributes_num++;
                }
            } else if (s.equals("@data")) {
                data_started_ = true;
            }
        }
        s = parent_.get();
        while (s != null && s.isEmpty()) { // Skip blanks.
            s = parent_.get();
        }

        if (s == null) { // End of data stream.
            dataset_descriptor_.instances_num--;
            return null;
        }

        dataset_descriptor_.instances_num++;
        return Util.parse(s, ",");
    }

    /**
     * Returns the constructed dataset descriptor.
     * It should be accessed only when the whole stream was read.
     */
    public DatasetDescriptor getDatasetDescriptor() {
        return dataset_descriptor_;
    }

    /**
     * Resets the parent and the dataset descriptor.
     */
    @Override
    public void reset() {
        data_started_ = false;
        parent_.reset();
    }

    @Override
    public Parser clone() {
        return new Parser(parent_);
    }
}
