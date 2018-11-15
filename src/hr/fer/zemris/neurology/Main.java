package hr.fer.zemris.neurology;

import hr.fer.zemris.data.*;
import hr.fer.zemris.data.datasets.BinaryDecoderClassification;
import hr.fer.zemris.data.datasets.ComplexFunctionRegression;
import hr.fer.zemris.data.modifiers.IModifier;
import hr.fer.zemris.data.modifiers.Randomizer;
import hr.fer.zemris.data.primitives.DataPair;
import hr.fer.zemris.data.primitives.TensorPair;
import hr.fer.zemris.neurology.descendmethods.AdamOptimizer;
import hr.fer.zemris.neurology.descendmethods.MomentumOptimizer;
import hr.fer.zemris.tf.TFContext;
import hr.fer.zemris.tf.TFStep;
import hr.fer.zemris.tf.Utils;
import org.tensorflow.*;
import org.tensorflow.Shape;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.*;

import javax.xml.crypto.Data;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;


public class Main {
    private static DatasetDescriptor descriptor;
    private static Cacher cacher;

    public static void main(String[] args) {
        // Construct dataset pipeline.
        APipe<?, DataPair> dataset = new BinaryDecoderClassification();
        cacher = new Cacher(dataset, new IModifier[]{new Randomizer(1)});
        descriptor = ((IDescriptableDS) dataset).describe();
        Tensorifyer<Float> pipeline = new Tensorifyer(new Batcher(cacher, 3));
        // Run the nn procedures.
        try (Graph g = new Graph()) {
            try (Session sess = new Session(g)) {
                NN net = new NN(g, sess);
                net.build();
                net.initialize();
                System.out.println("Training...");
                net.train(pipeline);
                System.out.println("Predicting...");
                net.predict(pipeline);
            }
        }
    }

    @SuppressWarnings("Duplicates")
    static class NN implements INeuralNetwork<TensorPair<Float>> {
        private Graph graph;
        private Ops tf;
        private Session sess;

        private FullyConnectedLayer<Float>[] layers;

        private Placeholder<Float> x, y;
        private Operand<Float> logits;
        private Operand<Float> loss;
        private Operand<Long> output;
        private Operand<Float> accuracy;

        public NN(Graph g, Session session) {
            graph = g;
            tf = Ops.create(graph);
            sess = session;

            IActivationFunction<Float> af = input -> tf.sigmoid(input);

            layers = new FullyConnectedLayer[]{
                    new FullyConnectedLayer(tf, 5, af, Float.class),
                    new FullyConnectedLayer(tf, 5, af, Float.class),
                    new FullyConnectedLayer(tf, descriptor.getClassesNumber(), null, Float.class)
            };
        }

        @Override
        public void build() {
            // Define inputs.
            x = tf.placeholder(Float.class, Placeholder.shape(Shape.make(-1, descriptor.getAttributesNumber())));
            y = tf.placeholder(Float.class, Placeholder.shape(Shape.make(-1, descriptor.getClassesNumber())));
            // Build layers.
            logits = x;
            for (ILayer<Float> l : layers)
                logits = l.build(logits);
            logits = tf.softmax(logits);
            output = tf.argMax(logits, tf.constant(1));
            accuracy = tf.reduceSum(tf.cast(tf.equal(tf.cast(y, Long.class), output), Float.class), tf.constant(0));
            // Define loss.
//            loss = tf.pow(tf.sub(logits, y), tf.constant(2f));
            loss = tf.reduceMean(tf.reduceSum(tf.neg(tf.mul(
                    tf.cast(tf.oneHot(tf.cast(y, Integer.class), tf.constant(descriptor.getClassesNumber()), tf.constant(1), tf.constant(0)), Float.class),
                    tf.log(logits)
            )), tf.constant(1)), tf.constant(0));
        }

        @Override
        public void initialize() {
            long seed = 42;
            TFContext context = new TFContext();
            // Construct layer initialization.
            for (ILayer l : layers)
                l.initialize(context, seed);
            // Run initialization step.
            try (TFStep initStep = context.createStep(sess)) {
                initStep.run();
            }
        }

        @Override
        public void train(APipe<?, TensorPair<Float>> dataset) {
            TFContext context = new TFContext();
            // Register layer gradients.
            List<Operand<Float>> grad_list = new LinkedList<>();
            for (ILayer<Float> l : layers)
                l.registerGradients(grad_list);
            Gradients gradients = tf.gradients(loss, grad_list);
            // Define descend method.
            Constant<Float> learning_rate = tf.constant(5e-1f);
            Constant<Float> momentum = tf.constant(1e-1f);
            for (int i = 0; i < grad_list.size(); i++) {
                context.addTarget(new MomentumOptimizer<>(learning_rate, momentum).apply(
                        tf, context, grad_list.get(i), gradients.dy(i), Float.class
                ));
            }
            // Add fetchables.
            context.addTargetToFetch(loss);
            context.addTargetToFetch(accuracy);

            // Training procedure.
            for (int epoch = 0; epoch < 100000; epoch++) {
                cacher.applyModifier(new Randomizer(2)); // Randomize inputs (2 passes).
                double lss = 0;
                double acc = 0;
                int ctr = 0;
                dataset.reset();
                for (TensorPair<Float> tp = dataset.get(); tp != null; tp = dataset.get(), ctr++) {
                    try (TFStep step = context.createStep(sess)) {
                        step
                                .feed(x, tp.getKey())
                                .feed(y, tp.getVal())
                                .run();
                        // Fetch loss.
                        FloatBuffer fb = FloatBuffer.allocate(tp.getVal().numElements());
                        step.target(loss).writeTo(fb);
                        lss += fb.get(0);
                        acc += step.target(accuracy).floatValue();
                    }
                }
                lss /= ctr;
                if (lss < 1e-6) {
                    System.out.println("Goal reached: " + lss);
                    break;
                }
                if (epoch % 100 == 0) {
                    System.out.println("Epoch " + (epoch + 1) + " has accuracy: " + lss + "   (" + acc + " correct)");
                }
            }
        }

        @Override
        public TensorPair<Float>[] predict(APipe<?, TensorPair<Float>> dataset) {
            TFContext context = new TFContext();
            // Add fetchables.
            context.addTargetToFetch(output);
            context.addTargetToFetch(accuracy);

            // Print predictions.
            dataset.reset();
            int corr = 0;
            LinkedList<TensorPair<Float>> predictions = new LinkedList<>();
            for (TensorPair<Float> tp = dataset.get(); tp != null; tp = dataset.get()) {
                try (TFStep step = context.createStep(sess)) {
                    step
                            .feed(x, tp.getKey())
                            .run();
                    // Print out predictions and number of hits.
                    System.out.println(Utils.toString(tp.getKey(), Float.class) + " --> "
                            + Utils.toString(step.target(output), Long.class));
                    corr += step.target(accuracy).floatValue();
                    predictions.add(new TensorPair(null, step.target(output)));
                }
            }
            System.out.println("Correct: " + corr);
            return predictions.toArray(new TensorPair[]{});
        }
    }
}
