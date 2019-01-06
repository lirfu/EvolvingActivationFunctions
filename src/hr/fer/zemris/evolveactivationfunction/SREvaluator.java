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

public class SREvaluator extends AEvaluator<DerivableSymbolicTree> {
    private TrainProcedure tmpl_procedure_;
    private int[] architecture_;
    private ILogger log_;

    public SREvaluator(TrainProcedure template_procedure, int[] architecture, ILogger log) {
        tmpl_procedure_ = template_procedure;
        architecture_ = architecture;
        log_ = log;
    }

    @Override
    public double performEvaluate(DerivableSymbolicTree g) {
        IActivation[] activations = new IActivation[architecture_.length];
        for (int i = 0; i < activations.length; i++)
            activations[i] = new CustomFunction(g.copy());

        System.out.println("Evaluating: " + g.serialize());
        CommonModel model = tmpl_procedure_.createModel(architecture_, activations);

        tmpl_procedure_.train(model, log_, null);
        Pair<ModelReport, INDArray> res = tmpl_procedure_.test(model);

        return res.getKey().f1();
    }
}
