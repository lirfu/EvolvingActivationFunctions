package hr.fer.zemris.evolveactivationfunction.nn;

import hr.fer.zemris.evolveactivationfunction.Context;
import hr.fer.zemris.neurology.dl4j.ModelReport;
import hr.fer.zemris.utils.Pair;
import hr.fer.zemris.utils.logs.ILogger;
import org.deeplearning4j.api.storage.StatsStorageRouter;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;
import org.nd4j.linalg.activations.IActivation;

import java.io.IOException;

public interface ITrainProcedure {
    public String describeDatasets();

    public Context createContext(String experiment_name);

    public IModel createModel(NetworkArchitecture architecture, IActivation[] activations);

    public void train(@NotNull IModel model, @NotNull ILogger log, @Nullable StatsStorageRouter stats_storage);

    public Pair<ModelReport, Object> test(@NotNull IModel model);

    Pair<ModelReport, Object> createAndRun(NetworkArchitecture architecture, IActivation[] activations, @NotNull ILogger log, @Nullable StatsStorageRouter stats_storage);

    public void storeResults(IModel model, Context context, Pair<ModelReport, Object> result) throws IOException;
}
