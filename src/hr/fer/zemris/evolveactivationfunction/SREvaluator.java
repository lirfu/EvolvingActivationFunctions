package hr.fer.zemris.evolveactivationfunction;

import hr.fer.zemris.evolveactivationfunction.activationfunction.CustomFunction;
import hr.fer.zemris.evolveactivationfunction.activationfunction.DerivableSymbolicTree;
import hr.fer.zemris.genetics.AEvaluator;
import hr.fer.zemris.genetics.symboregression.SymbolicTree;
import hr.fer.zemris.neurology.dl4j.ModelReport;
import hr.fer.zemris.utils.Pair;
import hr.fer.zemris.utils.logs.ILogger;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.IOException;
import java.util.HashMap;
import java.util.LinkedList;

public class SREvaluator extends AEvaluator<DerivableSymbolicTree> {
    private TrainProcedure tmpl_procedure_;
    private int[] architecture_;
    private ILogger log_;

    private boolean use_memory_;
    private HashMap<String, Double> memory = new HashMap<>();

    public SREvaluator(TrainProcedure template_procedure, int[] architecture, ILogger log) {
        tmpl_procedure_ = template_procedure;
        architecture_ = architecture;
        log_ = log;
    }

    public SREvaluator(TrainProcedure template_procedure, int[] architecture, ILogger log, boolean use_memory) {
        this(template_procedure, architecture, log);
        use_memory_ = use_memory;
    }

    @Override
    public double performEvaluate(DerivableSymbolicTree g) {
        String s = g.serialize();
        Double fitness;
        if (use_memory_ && (fitness = memory.get(s)) != null) {
            return fitness;
        }

        IActivation[] activations = new IActivation[architecture_.length];
        for (int i = 0; i < activations.length; i++)
            activations[i] = new CustomFunction(g.copy());

        System.out.println("Evaluating: " + s);
        CommonModel model = tmpl_procedure_.createModel(architecture_, activations);

        tmpl_procedure_.train(model, log_, null);
        Pair<ModelReport, INDArray> res = tmpl_procedure_.test(model);

        // TODO Store ot write.

        return res.getKey().f1();
    }
}
