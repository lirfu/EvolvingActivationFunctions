package hr.fer.zemris.neurology.dl4j;

import hr.fer.zemris.utils.logs.ILogger;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

public interface IModel {
    /**
     * Trains the model on given dataset.
     *
     * @param trainset Dataset for training.
     * @param log      Logger for training messages.
     */
    public void train(@NotNull DataSetIterator trainset, @NotNull ILogger log);

    /**
     * Tests the model on given dataset.
     *
     * @param testset Dataset for testing.
     * @param log     Logger for testing messages.
     */
    public void test(@NotNull DataSetIterator testset, @NotNull ILogger log, @Nullable IReport report);

    /**
     * Predicts outputs for given inputs.
     *
     * @param inputs Set of inputs.
     * @param log    Logger for prediction messages.
     */
    public void predict(@NotNull DataSetIterator inputs, @NotNull ILogger log);
}
