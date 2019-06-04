package hr.fer.zemris.evolveactivationfunction.nn;

import hr.fer.zemris.evolveactivationfunction.Context;
import hr.fer.zemris.evolveactivationfunction.EvolvingActivationParams;
import hr.fer.zemris.neurology.dl4j.ModelReport;
import hr.fer.zemris.utils.Pair;
import hr.fer.zemris.utils.PythonBridge;
import hr.fer.zemris.utils.logs.ILogger;
import org.apache.commons.lang.NotImplementedException;
import org.deeplearning4j.api.storage.StatsStorageRouter;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;
import org.nd4j.linalg.activations.IActivation;

import java.io.IOException;

public class TrainProcedureTFPy implements ITrainProcedure {
    private static final String PYTHON_SCRIPT = "/python/model_evaluator.py";
    private EvolvingActivationParams params_;
    private PythonBridge.Session bridge_;

    public TrainProcedureTFPy(EvolvingActivationParams params) throws IOException {
        params_ = params;
        bridge_ = new PythonBridge(PYTHON_SCRIPT).openSession();
    }

    @Override
    public String describeDatasets() {
        try {
            TrainProcedureDL4J tp = new TrainProcedureDL4J(params_);  // #lazyness
            return tp.describeDatasets();
        } catch (IOException | InterruptedException e) {
            e.printStackTrace();
        }
        return null;
    }

    @Override
    public Context createContext(String experiment_name) {
        return new Context(params_.name(), experiment_name);
    }

    @Override
    public IModel createModel(NetworkArchitecture architecture, IActivation[] activations) {
        throw new NotImplementedException("This method is not used!");
    }

    @Override
    public void train(@NotNull IModel model, @NotNull ILogger log, @Nullable StatsStorageRouter stats_storage) {
        throw new NotImplementedException("This method is not used!");
    }

    @Override
    public void train_joined(@NotNull IModel model, @NotNull ILogger log, @Nullable StatsStorageRouter stats_storage) {

    }

    @Override
    public Pair<ModelReport, Object> validate(@NotNull IModel model) {
        return null;
    }

    @Override
    public Pair<ModelReport, Object> test(@NotNull IModel model) {
        throw new NotImplementedException("This method is not used!");
    }

    public Pair<ModelReport, Object> createAndRun(NetworkArchitecture architecture, IActivation[] activations, @NotNull ILogger log, @Nullable StatsStorageRouter stats_storage) {
        try {
            //        bridge_.write(string_neki); // TODO Send commands to run session.


            String s;
            while (!(s = bridge_.read()).equals("done")) {
                log.d(s); // Log procedure outputs
            }
            // Return deserialized results.
            s = bridge_.read(); // Read results.
            ModelReport mr = new ModelReport();
            mr.parse(s);
            return new Pair<>(mr, null);
        } catch (IOException e) {
            e.printStackTrace();
            return null;
        }
    }

    @Override
    public void storeResults(IModel model, Context context, Pair<ModelReport, Object> result) throws IOException {
//TODO
    }

    @Override
    protected void finalize() throws Throwable {
        super.finalize();
        bridge_.close();
    }
}
