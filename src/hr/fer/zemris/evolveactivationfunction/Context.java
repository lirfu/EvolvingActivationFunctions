package hr.fer.zemris.evolveactivationfunction;

import org.deeplearning4j.ui.api.UIServer;
import org.jetbrains.annotations.NotNull;

/**
 * Defines the dataset name and the experiment name. Used to uniquely identify the folder containing the experiment and its results.
 */
public class Context {
    private String dataset_name_;
    private String experiment_name_;

    /**
     * Creates a context for this experiment.
     *
     * @param dataset_name    Name of the dataset used.
     * @param experiment_name Name of the experiment.
     */
    public Context(@NotNull String dataset_name, @NotNull String experiment_name) {
        dataset_name_ = dataset_name;
        experiment_name_ = experiment_name;
    }

    public String getDatasetName() {
        return dataset_name_;
    }

    public String getExperimentName() {
        return experiment_name_;
    }
}
