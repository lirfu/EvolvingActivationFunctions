package hr.fer.zemris.evolveactivationfunction.programs;

import java.io.IOException;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

@SuppressWarnings("ALL")
public class WorkerProgram {
    public static void main(String[] args) throws IOException, InterruptedException {
        String ARRAY_REGEX = "\\{ *(.*) *\\}";

        String s = "aaa {fc(1), conv(2) , a(3)} ";

        Pattern p = Pattern.compile(ARRAY_REGEX);
        Matcher m = p.matcher(s);

        if (m.find()) {
            System.out.println(m.groupCount());
            for (int i = 1; i <= m.groupCount(); i++)
                System.out.println(i + ": " + m.group(i));
        }

//        // Set double precision globally.
//        Nd4j.setDataType(DataBuffer.Type.DOUBLE);
//        EvolvingActivationParams.initialize(new ISerializable[]{});
//
//        // Initialize the environment.
//        CustomFunction activation = new CustomFunction(new DerivableSymbolicTree(DerivableSymbolicTree.parse(
//                "sin[*[-[x,0.7580365993685689],0.7580365993685689]]",
//                TreeNodeSetFactory.build(new Random(), TreeNodeSets.ALL)
//        )));
//        EvolvingActivationParams params = StorageManager.loadEvolutionParameters(
//                "sol/noiseless_all_training_9class/03_fixed_algorithms_train80/evolution_parameters.txt"
//        );
//        params.parse("epochs_num    1");
//        TrainProcedure proc = new TrainProcedure(params);
//        Context c = proc.createContext("00_test_model_saving");
//
//        // Training and storing the model.
//        System.out.println("Creating model...");
//        CommonModel model = proc.createModel(new NetworkArchitecture("fc(30)-fc(30)"), new IActivation[]{activation, activation});
//        System.out.println("Training model...");
//        proc.train(model, new StdoutLogger(), null);
//        System.out.println("Store model...");
////        for (int i = 0; i < model.getModel().getnLayers(); i++)
////            System.out.println(i + " --> " + model.getModel().getLayer(i).conf().toJson());
//        StorageManager.storeModel(model, c);
//
//        // Loading and testing the model.
//        System.out.println("Load model...");
//        CommonModel m = StorageManager.loadModel(c);
//        System.out.println("Testing...");
//        Pair<ModelReport, INDArray> res = proc.test(m);
//        System.out.println(res.getKey());
    }
}
