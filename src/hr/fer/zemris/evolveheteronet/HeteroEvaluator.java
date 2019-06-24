package hr.fer.zemris.evolveheteronet;

import hr.fer.zemris.evolveactivationfunction.nn.CustomFunction;
import hr.fer.zemris.evolveactivationfunction.nn.IModel;
import hr.fer.zemris.evolveactivationfunction.nn.NetworkArchitecture;
import hr.fer.zemris.evolveactivationfunction.nn.TrainProcedureDL4J;
import hr.fer.zemris.evolveactivationfunction.tree.DerivableSymbolicTree;
import hr.fer.zemris.evolveactivationfunction.tree.nodes.InputNode;
import hr.fer.zemris.genetics.AEvaluator;
import hr.fer.zemris.genetics.symboregression.TreeNode;
import hr.fer.zemris.genetics.symboregression.TreeNodeSet;
import hr.fer.zemris.genetics.vector.intvector.IntVectorGenotype;
import hr.fer.zemris.neurology.dl4j.ModelReport;
import hr.fer.zemris.utils.Pair;
import hr.fer.zemris.utils.Stopwatch;
import hr.fer.zemris.utils.Utilities;
import hr.fer.zemris.utils.logs.ILogger;
import org.deeplearning4j.api.storage.StatsStorageRouter;
import org.nd4j.linalg.activations.IActivation;

import java.util.HashMap;

public class HeteroEvaluator extends AEvaluator<IntVectorGenotype> {
    private TrainProcedureDL4J tmpl_procedure_;
    private NetworkArchitecture architecture_;
    private ILogger log_;
    private TreeNodeSet set_;

    private boolean use_memory_;
    private final HashMap<String, Double> memory = new HashMap<>();

    public HeteroEvaluator(TrainProcedureDL4J template_procedure, NetworkArchitecture architecture, ILogger log, TreeNodeSet set) {
        tmpl_procedure_ = template_procedure;
        architecture_ = architecture;
        log_ = log;
        set_ = set;
    }

    public HeteroEvaluator(TrainProcedureDL4J template_procedure, NetworkArchitecture architecture, ILogger log, TreeNodeSet set, boolean use_memory) {
        this(template_procedure, architecture, log, set);
        use_memory_ = use_memory;
    }

    private String stringify(IntVectorGenotype g) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < g.size(); i++) {
            if (i > 0)
                sb.append('-');
            sb.append(set_.getNode(g.get(i)).getName());
        }
        return sb.toString();
    }

    @Override
    public double performEvaluate(IntVectorGenotype g) {
        String s = stringify(g);
        Double fitness;
        if (use_memory_ && (fitness = memory.get(s)) != null) {
            log_.i("Re-using stored fitness for: " + s);
            return fitness;
        }

        log_.i("Evaluating: " + s);
        Pair<ModelReport, Object> res = evaluateModel(g, null, s);

        fitness = -res.getKey().f1(); // Negative is for minimization.

        if (!Double.isFinite(fitness)) {
            fitness = 0.;
        }

        if (use_memory_) {
            synchronized (memory) {
                memory.put(s, fitness);
            }
        }

        return fitness;
    }

    public IModel buildModelFrom(IntVectorGenotype g) {
        IActivation[] activations = new IActivation[architecture_.layersNum()];
        for (int i = 0; i < activations.length; i++) {
            TreeNode root = set_.getNode(g.get(i));
            if (root.getChildrenNum() == 1)
                root.getChildren()[0] = new InputNode();
            activations[i] = new CustomFunction(new DerivableSymbolicTree(set_, root));
        }
        return tmpl_procedure_.createModel(architecture_, activations);
    }

    public Pair<ModelReport, Object> evaluateModel(IntVectorGenotype g, StatsStorageRouter storage, String name) {
        Stopwatch timer = new Stopwatch();
        timer.start();

        IModel model = buildModelFrom(g);
        tmpl_procedure_.train_itersearch(model, log_, storage);
        Pair<ModelReport, Object> res = tmpl_procedure_.validate(model);

        log_.i("(" + Utilities.formatMiliseconds(timer.stop()) + ") Done evaluating: " + name + "(" + res.getKey().f1() + ")");
        return res;
    }
}
