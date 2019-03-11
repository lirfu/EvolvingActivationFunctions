package hr.fer.zemris.evolveactivationfunction;


import hr.fer.zemris.data.Parser;
import hr.fer.zemris.data.Reader;
import hr.fer.zemris.data.UnsafeDatasetDescriptor;
import hr.fer.zemris.evolveactivationfunction.nn.CommonModel;
import hr.fer.zemris.neurology.dl4j.TrainParams;
import hr.fer.zemris.neurology.dl4j.ModelReport;
import hr.fer.zemris.utils.logs.FileLogger;
import hr.fer.zemris.utils.logs.ILogger;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.ui.storage.FileStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.*;

/**
 * Manages the structure of resources and solution.
 * <p>Structure of solutions:</p>
 * <p><code>
 * sol<br>
 * - dataset1_name<br>
 * - - experiment1_name<br>
 * - - - - model.zip<br>
 * - - - - parameters.txt<br>
 * - - - - predictions.txt<br>
 * - - - - results.txt<br>
 * - - - - train.log<br>
 * - - experiment2_name<br>
 * - - - - model.zip<br>
 * - - - - parameters.txt<br>
 * - - - - predictions.txt<br>
 * - - - - results.txt<br>
 * - - - - train.log<br>
 * - - ...<br>
 * - dataset2_name<br>
 * - - ...<br>
 * ...
 * </code></p>
 */
public class StorageManager {
    public static final String current_dir_ = System.getProperty("user.dir") + File.separator;
    public static final String res_dir_name_ = current_dir_ + "res" + File.separator;
    public static final String sol_dir_name_ = current_dir_ + "sol" + File.separator;
    public static final String tmp_dir_name_ = current_dir_ + "tmp" + File.separator;
    public static final String sol_model_name_ = "model.zip";
    public static final String sol_predictions_name_ = "predictions.txt";
    public static final String sol_result_name_ = "results.txt";
    public static final String sol_stats_name_ = "stats.dl4jlog";
    public static final String sol_train_log_name_ = "train.log";
    public static final String sol_train_params_name_ = "train_parameters.txt";
    public static final String sol_evo_log_name_ = "evolution.log";
    public static final String sol_evo_params_name_ = "evolution_parameters.txt";
    public static final String sol_best_func_name_ = "best_function.png";
    public static final String sol_top_func_name_ = "top_functions.png";

    static {
        File sol = new File(sol_dir_name_);
        if (!sol.exists() && !sol.mkdirs()) {
            throw new RuntimeException("Cannot create solution directory: " + sol_model_name_);
        }
        File tmp = new File(tmp_dir_name_);
        if (!tmp.exists() && !tmp.mkdirs()) {
            throw new RuntimeException("Cannot create temporary directory: " + tmp_dir_name_);
        }
    }

    private static File createFile(String path) throws IOException {
        File f = new File(path);
        if (!f.exists() && !f.getParentFile().mkdirs() && !f.createNewFile()) {
            throw new IOException("Cannot create file: " + f);
        }
        return f;
    }

    private static String readEntireFile(String path) throws IOException {
        BufferedReader reader = new BufferedReader(new FileReader(path));
        StringBuilder sb = new StringBuilder();
        String s;
        while ((s = reader.readLine()) != null) {
            sb.append(s).append('\n');
        }
        reader.close();
        return sb.toString();
    }

    public static String createExperimentPath(Context c) {
        return sol_dir_name_ + c.getDatasetName() + File.separator + c.getExperimentName() + File.separator;
    }

    public static String getResourcesPath() {
        return res_dir_name_;
    }

    public static String getSolutionsPath() {
        return sol_dir_name_;
    }

    public static String getTemporaryDirPath() {
        return tmp_dir_name_;
    }


    /* SOLUTION FOLDER MANAGEMENT */

    /**
     * Creates a file logger to log the training process.
     */
    public static ILogger createTrainingLogger(Context c) throws IOException {
        return new FileLogger(createExperimentPath(c) + sol_train_log_name_, true);
    }

    /**
     * Creates a file logger to log the training process.
     */
    public static ILogger createEvolutionLogger(Context c) throws IOException {
        return new FileLogger(createExperimentPath(c) + sol_evo_log_name_, true);
    }

    /**
     * Creates the storage for training statistics.
     */
    public static FileStatsStorage createStatsLogger(Context c) {
        return new FileStatsStorage(new File(createExperimentPath(c) + sol_stats_name_));
    }

    /**
     * Stores the solution model of the given context.
     */
    public static void storeModel(CommonModel model, Context c) throws IOException {
        String path = createExperimentPath(c) + sol_model_name_;
        File f = createFile(path);
        ModelSerializer.writeModel(model.getModel(), f, true);
    }

    /**
     * Loads the solution model of the given context.
     *
     * @throws IOException - if the model file cannot be read or doesn't exist.
     */
    public static CommonModel loadModel(Context c) throws IOException {
        return new CommonModel(ModelSerializer.restoreMultiLayerNetwork(createExperimentPath(c) + sol_model_name_));
    }

    /**
     * Stores the predictions of the given context.
     */
    public static void storePredictions(INDArray predictions, Context c) throws IOException {
        Nd4j.writeTxt(predictions, createExperimentPath(c) + sol_predictions_name_);
    }

    /**
     * Load the predictions of the given context.
     */
    public static INDArray loadPredictions(Context c) throws IOException {
        return Nd4j.readTxt(createExperimentPath(c) + sol_predictions_name_);
    }

    /**
     * Stores the parameters used in given context.
     */
    public static void storeTrainParameters(TrainParams params, Context c) throws IOException {
        FileLogger log = new FileLogger(createExperimentPath(c) + sol_train_params_name_, false);
        log.o(params.serialize());
    }

    /**
     * Loads the parameters used in given context.
     */
    public static TrainParams loadTrainParameters(Context c) throws IOException {
        TrainParams params = new TrainParams();
        for (String line : readEntireFile(createExperimentPath(c) + sol_train_params_name_).split("\n"))
            if (!line.isEmpty())
                params.parse(line);
        return params;
    }

    /**
     * Stores evolution parameters used in given context.
     */
    public static void storeEvolutionParams(EvolvingActivationParams params, Context c) throws IOException {
        FileLogger log = new FileLogger(createExperimentPath(c) + sol_evo_params_name_, false);
        log.o(params.serialize());
    }

    /**
     * Loads the parameters used in given context.
     */
    public static EvolvingActivationParams loadEvolutionParameters(String path) throws IOException {
        EvolvingActivationParams params = new EvolvingActivationParams();
        for (String line : readEntireFile(path).split("\n"))
            if (!line.isEmpty())
                params.parse(line);
        return params;
    }

    /**
     * Stores the resulting statistics of the given context.
     */
    public static void storeResults(ModelReport report, Context c) throws IOException {
        FileLogger log = new FileLogger(createExperimentPath(c) + sol_result_name_, false);
        log.d(report.toString());
    }

    /**
     * Loads the resulting statistics of the given context.
     */
    public static ModelReport loadResults(Context c) throws IOException {
        ModelReport report = new ModelReport();
        report.parse(readEntireFile(createExperimentPath(c) + sol_result_name_));
        return report;
    }

    /**
     * Writes the image to a png file.
     */
    public static void writeImageOfBest(BufferedImage img, Context c) throws IOException {
        File f = createFile(createExperimentPath(c) + sol_best_func_name_);
        ImageIO.write(img, "png", f);
    }

    /**
     * Writes the image to a png file.
     */
    public static void writeImageOfTop(BufferedImage img, Context c) throws IOException {
        File f = createFile(createExperimentPath(c) + sol_top_func_name_);
        ImageIO.write(img, "png", f);
    }


    /* RESOURCE FOLDER MANAGEMENT */

    /**
     * Loads the entire arff dataset. The file is first parsed with custom .arff parser and then read using DL4Js' CSVRecordReader.
     *
     * @param dataset_path File path of the dataset.
     * @return The entire dataset.
     */
    public static DataSet loadEntireArffDataset(String dataset_path) throws IOException, InterruptedException {
        Parser p = new Parser(new Reader(dataset_path));
        while (p.next() != null) ;  // Read all to generate descriptor.
        UnsafeDatasetDescriptor desc = p.getDatasetDescriptor();
        // Load the dataset.
        CSVRecordReader set = new CSVRecordReader(desc.skip_lines, ',');
        set.initialize(new FileSplit(new File(dataset_path)));
        return new RecordReaderDataSetIterator(set, desc.instances_num, desc.attributes_num, desc.classes_num).next();
    }

    public static DataSet loadEntireCsvDataset(String dataset_path) throws IOException, InterruptedException {
        CSVRecordReader set = new CSVRecordReader(',');
        set.initialize(new FileSplit(new File(dataset_path)));

        int instances_num = 0;
        while (set.hasNext()) {
            set.next();
            instances_num++;
        }
        set.reset();

        return new RecordReaderDataSetIterator(set, instances_num).next();
    }

    /**
     * Extract the snake-case dataset name from its path. Removes the extension and the last word separated by an underscore (train/test).
     * <p>E.g. "res/noiseless/5k/noiseless_9class_5k_train.arff" -> "noiseless_9class_5k"</p>
     *
     * @param dataset_path Path to the dataset name.
     * @return Snake-case string representing the dataset name.
     */
    public static String dsNameFromPath(String dataset_path, boolean remove_last_underscore) {
        String n = new File(dataset_path).getName();
        n = n.substring(0, n.lastIndexOf('.'));
        if (remove_last_underscore)
            return n.replaceAll("_[^_]+$", "");
        return n;
    }
}
