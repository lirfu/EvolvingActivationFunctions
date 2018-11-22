package hr.fer.zemris.neurology.demo;

import hr.fer.zemris.data.*;
import hr.fer.zemris.data.datasets.BinaryDecoderClassification;
import hr.fer.zemris.data.modifiers.IModifier;
import hr.fer.zemris.data.modifiers.Randomizer;
import hr.fer.zemris.data.primitives.DataPair;
import hr.fer.zemris.data.primitives.TensorPair;
import hr.fer.zemris.neurology.tf.FullyConnectedLayer;
import hr.fer.zemris.neurology.tf.IActivationFunction;
import hr.fer.zemris.neurology.tf.ILayer;
import hr.fer.zemris.neurology.tf.INeuralNetwork;
import hr.fer.zemris.neurology.tf.descendmethods.AdamOptimizer;
import hr.fer.zemris.neurology.tf.TFContext;
import hr.fer.zemris.neurology.tf.TFStep;
import hr.fer.zemris.neurology.tf.Utils;
import org.tensorflow.*;
import org.tensorflow.Shape;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.*;

import java.nio.FloatBuffer;
import java.util.LinkedList;
import java.util.List;
import java.util.Random;


public class DemoTF {
    private static DatasetDescriptor descriptor;
    private static Modifier cacher;

    public static void main(String[] args) {
        // Construct dataset pipeline.
        APipe<?, DataPair> dataset = new BinaryDecoderClassification();
        cacher = new Modifier(dataset, new IModifier[]{});
        descriptor = ((IDescriptableDS) dataset).describe();
        Tensorifyer<Float> pipeline = new Tensorifyer(new Batcher(cacher, 4));
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

            IActivationFunction<Float> af = input -> tf.relu(input);

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
            Constant<Float> learning_rate = tf.constant(1e-4f);
            Constant<Float> momentum_rate = tf.constant(1e-2f);
            for (int i = 0; i < grad_list.size(); i++) {
                context.addTarget(new AdamOptimizer(learning_rate).apply(
                        tf, context, grad_list.get(i), gradients.dy(i), Float.class
                ));
            }
            // Add fetchables.
            context.addTargetToFetch(loss);
            context.addTargetToFetch(accuracy);

            Randomizer rand = new Randomizer(2, new Random(42));

            // Training procedure.
            float t = 0;
            for (int epoch = 0; epoch < 10000; epoch++) {
//                cacher.applyModifier(rand); // Randomize inputs (2 passes).
                double lss = 0;
                double acc = 0;
                int ctr = 0;
                t++;
                dataset.reset();
                for (TensorPair<Float> tp = dataset.next(); tp != null; tp = dataset.next(), ctr++) {
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
                if (Double.isNaN(lss)) {
                    System.out.println("Error! Loss is NaN!");
                    break;
                } else if (lss < 1e-6) {
                    System.out.println("Goal reached: " + lss);
                    break;
                } else if (epoch % 100 == 0) {
                    System.out.println("Epoch " + (epoch + 1) + " has accuracy: " + lss + "   (" + acc + " correct)");
                }
            }
        }

        @Override
        public TensorPair<Float>[] predict(APipe<?, TensorPair<Float>> dataset) {
            TFContext context = new TFContext();
            // Add fetchables.
            context.addTargetToFetch(output);

            // Print predictions.
            dataset.reset();
            LinkedList<TensorPair<Float>> predictions = new LinkedList<>();
            for (TensorPair<Float> tp = dataset.next(); tp != null; tp = dataset.next()) {
                try (TFStep step = context.createStep(sess)) {
                    step
                            .feed(x, tp.getKey())
                            .run();
                    // Print out predictions and number of hits.
                    System.out.println(Utils.toString(tp.getKey(), Float.class) + " --> "
                            + Utils.toString(step.target(output), Long.class));
                    predictions.add(new TensorPair(null, step.target(output)));
                }
            }
            return predictions.toArray(new TensorPair[]{});
        }
    }
}
