package hr.fer.zemris.neurology.dl4j;

import org.junit.Test;

import static org.junit.Assert.*;

public class TrainParamsTest {
    private interface IAction {
        boolean test();
    }

    @Test
    public void testParseSingle() {
        TrainParams p = new TrainParams();
        testSingle(p, "epochs_num", 42, () -> p.epochs_num() == 42);
        testSingle(p, "batch_size", 42, () -> p.batch_size() == 42);
        testSingle(p, "normalize_features", true, () -> p.normalize_features() == true);
        testSingle(p, "shuffle_batches", true, () -> p.shuffle_batches() == true);
        testSingle(p, "batch_norm", true, () -> p.batch_norm() == true);
        testSingle(p, "learning_rate", 0.01, () -> p.learning_rate() == 0.01);
        testSingle(p, "decay_rate", 0.01, () -> p.decay_rate() == 0.01);
        testSingle(p, "decay_step", 42, () -> p.decay_step() == 42);
        testSingle(p, "regularization_coef", 0.01, () -> p.regularization_coef() == 0.01);
        testSingle(p, "dropout_keep_prob", 0.01, () -> p.dropout_keep_prob() == 0.01);
        testSingle(p, "seed", 100L, () -> p.seed() == 100L);
        testSingle(p, "train_percentage", 0.01f, () -> p.train_percentage() == 0.01f);
    }

    private void testSingle(TrainParams p, String key, Object val, IAction a) {
        p.parse(key + "   " + val.toString());
        assertTrue("Parsing single value should work! (" + key + ")",
                a.test() && p.modifiable_params.size() == 0);
    }

    @Test
    public void testParseMulti() {
        TrainParams p = new TrainParams();
        testMultiple(p, "epochs_num", 42, 43, () -> p.modifiable_params.get(0).getVal().get(1).equals(43), 1);
        testMultiple(p, "batch_size", 42, 43, () -> p.modifiable_params.get(1).getVal().get(1).equals(43), 2);
        testMultiple(p, "normalize_features", true, false, () -> p.modifiable_params.get(2).getVal().get(1).equals(false), 3);
        testMultiple(p, "shuffle_batches", true, false, () -> p.modifiable_params.get(3).getVal().get(1).equals(false), 4);
        testMultiple(p, "batch_norm", true, false, () -> p.modifiable_params.get(4).getVal().get(1).equals(false), 5);
        testMultiple(p, "learning_rate", 0.01, 0.1, () -> p.modifiable_params.get(5).getVal().get(1).equals(0.1), 6);
        testMultiple(p, "decay_rate", 0.01, 0.1, () -> p.modifiable_params.get(6).getVal().get(1).equals(0.1), 7);
        testMultiple(p, "decay_step", 42, 43, () -> p.modifiable_params.get(7).getVal().get(1).equals(43), 8);
        testMultiple(p, "regularization_coef", 0.01, 0.1, () -> p.modifiable_params.get(8).getVal().get(1).equals(0.1), 9);
        testMultiple(p, "dropout_keep_prob", 0.01, 0.1, () -> p.modifiable_params.get(9).getVal().get(1).equals(0.1), 10);
        testMultiple(p, "seed", 100L, 10L, () -> p.modifiable_params.get(10).getVal().get(1).equals(10L), 11);
        testMultiple(p, "train_percentage", 0.01f, 0.1f, () -> p.modifiable_params.get(11).getVal().get(1).equals(0.1f), 12);
    }

    private void testMultiple(TrainParams p, String key, Object val1, Object val2, IAction a, int i) {
        p.parse(key + "   {" + val1.toString() + "  , " + val2.toString() + "}");
        assertTrue("Parsing multiple values should work! (" + key + ")",
                a.test() && p.modifiable_params.size() == i && p.modifiable_params.get(i - 1).getVal().size() == 2);
        System.out.println(p.modifiable_params.get(i - 1));
    }
}