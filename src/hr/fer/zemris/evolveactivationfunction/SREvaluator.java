package hr.fer.zemris.evolveactivationfunction;

import hr.fer.zemris.evolveactivationfunction.nn.*;
import hr.fer.zemris.evolveactivationfunction.tree.DerivableSymbolicTree;
import hr.fer.zemris.genetics.AEvaluator;
import hr.fer.zemris.neurology.dl4j.ModelReport;
import hr.fer.zemris.utils.Pair;
import hr.fer.zemris.utils.Stopwatch;
import hr.fer.zemris.utils.Utilities;
import hr.fer.zemris.utils.logs.ILogger;
import org.deeplearning4j.api.storage.StatsStorageRouter;
import org.nd4j.linalg.activations.IActivation;

import java.util.HashMap;

public class SREvaluator extends AEvaluator<DerivableSymbolicTree> {
    private TrainProcedureDL4J tmpl_procedure_;
    private NetworkArchitecture architecture_;
    private ILogger log_;
    private int max_depth_;
    private double default_max_value_;

    private boolean use_memory_;
    private final HashMap<String, Double> memory = new HashMap<>();

    public SREvaluator(TrainProcedureDL4J template_procedure, NetworkArchitecture architecture, ILogger log, int max_depth, double default_max_value) {
        tmpl_procedure_ = template_procedure;
        architecture_ = architecture;
        log_ = log;

        max_depth_ = max_depth;
        default_max_value_ = default_max_value;
    }

    public SREvaluator(TrainProcedureDL4J template_procedure, NetworkArchitecture architecture, ILogger log, int max_depth, double default_max_value, boolean use_memory) {
        this(template_procedure, architecture, log, max_depth, default_max_value);
        use_memory_ = use_memory;
    }

    @Override
    public double performEvaluate(DerivableSymbolicTree g) {
        Double fitness;
        String s = g.serialize();
        synchronized (memory) {
            if (use_memory_ && (fitness = memory.get(s)) != null) {
                log_.i("Re-using stored fitness for: " + s);
                return fitness;
            }
        }

        if (g.depth() > max_depth_) {
            log_.i("Tree too deep! Training skipped...");
            fitness = -1e-6;
        } else {
            log_.i("Evaluating: " + s);
            Pair<ModelReport, Object> res = evaluateModel(g, null, s);

            // Algorithm requires minimization in the negative fitness domain.
//        fitness = -res.getKey().f1();
            fitness = res.getKey().avg_guess_entropy() - default_max_value_ - 1e-6;

            if (!Double.isFinite(fitness)) {
                fitness = 0.;
            }
        }

        if (use_memory_) {
            synchronized (memory) {
                memory.put(s, fitness);
            }
        }

        return fitness;
    }

    public IModel buildModelFrom(DerivableSymbolicTree g) {
        IActivation[] activations = new IActivation[architecture_.layersNum()];
        for (int i = 0; i < activations.length; i++)
            activations[i] = new CustomFunction(g.copy());
        return tmpl_procedure_.createModel(architecture_, activations);
    }

    public Pair<ModelReport, Object> evaluateModel(DerivableSymbolicTree g, StatsStorageRouter storage, String name) {
        Stopwatch timer = new Stopwatch();
        timer.start();

        IModel model = buildModelFrom(g);
//        tmpl_procedure_.train(model, log_, storage);
        tmpl_procedure_.train_itersearch(model, log_, storage);
        Pair<ModelReport, Object> res = tmpl_procedure_.validate(model);

        log_.i("(" + Utilities.formatMiliseconds(timer.stop()) + ") Done evaluating: " + name + "(AGE: " + res.getKey().avg_guess_entropy() + ")");
        return res;
    }

    public ITrainProcedure getTrainProcedure() {
        return tmpl_procedure_;
    }
}
