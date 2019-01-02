package hr.fer.zemris.architecturesearch;

import hr.fer.zemris.evolveactivationfunction.Context;
import hr.fer.zemris.evolveactivationfunction.StorageManager;
import hr.fer.zemris.evolveactivationfunction.TrainProcedure;
import org.deeplearning4j.ui.storage.FileStatsStorage;

public class ViewExperiment {
    public static void main(String[] args) {
        String train_set_path = "res/noiseless/5k/256class/noiseless_256class_5k_train.arff";
        String experiment_name = "03_relu_30_30_overfit_normfeat_shflb";
        Context context = new Context(StorageManager.dsNameFromPath(train_set_path), experiment_name);
        FileStatsStorage stat_storage = StorageManager.createStatsLogger(context);
        TrainProcedure.displayTrainStats(stat_storage);
    }
}
