package hr.fer.zemris.evolveactivationfunction;

import hr.fer.zemris.utils.Pair;
import hr.fer.zemris.utils.logs.ILogger;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

import java.io.File;
import java.util.Iterator;
import java.util.LinkedList;

public class Datasets implements Iterable<Pair<String, String>> {
    private static volatile Datasets instance_;

    /**
     * Returns an instance of this class.
     * The given parameters are used only once (for initialization).
     * Subsequent calls ignore them.
     *
     * @param resource_dir_path Root folder of the datasets (or null).
     * @param logger            Logger to write errors to (or null).
     * @return Instance of this class with given parameters.
     */
    public static Datasets getInstance(@Nullable String resource_dir_path, @Nullable ILogger logger) {
        if (instance_ == null) {
            synchronized (instance_) {
                if (instance_ == null) {
                    instance_ = new Datasets(resource_dir_path, logger);
                }
            }
        }
        return instance_;
    }

    private ILogger logger_;
    private String resource_dir_path_;
    private Pair<String, String>[] datasets_;

    private Datasets(String resource_dir_path, ILogger logger) {
        logger_ = logger;
        resource_dir_path_ = resource_dir_path;
        datasets_ = getListOfDatasets();
    }

    /**
     * Returns a list of dataset filepath pairs.
     * Each pair is a train-test pair.
     * If a dataset has no pair, second element is <code>null</code>.
     */
    private Pair<String, String>[] getListOfDatasets() throws NullPointerException {
        LinkedList<Pair<String, String>> list = new LinkedList<>();
        String train, test;
        File res = new File(resource_dir_path_);
        for (File task : res.listFiles()) {  // DPAv2, DPAv4
            // TODO What's the data structure????
            for (File size : task.listFiles()) {  // 9k, 12k, ...
                for (File type : size.listFiles()) {  // noisy, noiseless, resampled...
                    train = type.getPath() + File.pathSeparator + "train.arff";
                    test = type.getPath() + File.pathSeparator + "test.arff";
                    if (!new File(train).exists()) {
                        logger_.logE("Can't find dataset: " + train);
                    } else if (!new File(test).exists()) {
                        list.add(new Pair<>(train, null));
                    } else {
                        list.add(new Pair<>(train, test));
                    }
                }
            }
        }
        return (Pair<String, String>[]) list.toArray();
    }

    /**
     * Iterates through train-test dataset pairs.
     * First elements are always defined.
     * Second elements might not be defined (if they don't exist).
     */
    @NotNull
    @Override
    public Iterator<Pair<String, String>> iterator() {
        return new Iterator<Pair<String, String>>() {
            private int index_ = 0;

            @Override
            public boolean hasNext() {
                return index_ >= datasets_.length;
            }

            @Override
            public Pair<String, String> next() {
                return new Pair<>(datasets_[index_++]);
            }
        };
    }
}
