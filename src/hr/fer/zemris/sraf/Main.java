package hr.fer.zemris.sraf;

import hr.fer.zemris.data.*;
import hr.fer.zemris.data.modifiers.IModifier;
import hr.fer.zemris.utils.Pair;
import hr.fer.zemris.utils.Stopwatch;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Arrays;


public class Main {
    public static void main(String[] args) throws IOException {
//        test_reader();
//        test_parser();
        test_cacher();
//        test_batcher();
    }

    public static void test_reader() throws FileNotFoundException {
        String path = "res/DPAv2_256/noisy_256class_1k_train.arff";
        Reader reader = new Reader(path);
        int i = 0;
        String s;
        while (i++ < 5 && (s = reader.get()) != null) {
            System.out.println(s);
        }
    }

    public static void test_parser() throws FileNotFoundException {
        String path = "res/DPAv2_256/noisy_256class_1k_train.arff";
        Parser parser = new Parser(new Reader(path));
        int i = 0;
        Pair<float[], Float> s;
        while (i++ < 5 && (s = parser.get()) != null) {
            System.out.println(Arrays.toString(s.getKey()) + " -- " + s.getVal());
        }
        System.out.println();
        System.out.println(parser.getDatasetDescriptor());
    }

    @SuppressWarnings("Duplicates")
    public static void test_cacher() throws FileNotFoundException {
        String path = "res/noisy/30k/noisy_9class_30k.arff";
        IModifier[] modifiers = new IModifier[]{};
        Stopwatch stopwatch = new Stopwatch();
        Pair<float[], Float> p;

        // Test loading time and memory consumption.
        long memoryS, memoryE;
        stopwatch.start();
        memoryS = Runtime.getRuntime().freeMemory();
        Parser parser = new Parser(new Reader(path));
        memoryE = Runtime.getRuntime().freeMemory();
        System.out.println("Loading parser: " + stopwatch.stop() + "ms\n                " + (memoryS - memoryE) + "B");
        stopwatch.start();
        memoryS = Runtime.getRuntime().freeMemory();
        Cacher cacher = new Cacher(new Parser(new Reader(path)), modifiers);
        memoryE = Runtime.getRuntime().freeMemory();
        System.out.println("Loading cacher: " + stopwatch.stop() + "ms\n                " + (memoryS - memoryE) + "B");

        // Test iteration time.
        int repeats = 100;
        float f = 0;
        long t = 0;
        for (int i = 0; i < repeats; i++) {
            f = 0;
            stopwatch.start();
            while ((p = parser.get()) != null) f += p.getVal();
            parser.reset();
            t += stopwatch.stop();
        }
        System.out.println("Iterating parser (avg): " + t / (double) repeats + "ms");
        System.out.println("Calculated value: " + f);

        t = 0;
        for (int i = 0; i < repeats; i++) {
            f = 0;
            stopwatch.start();
            while ((p = cacher.get()) != null) f += p.getVal();
            cacher.reset();
            t += stopwatch.stop();
        }
        System.out.println("Iterating cacher (avg): " + t / (double) repeats + "ms");
        System.out.println("Calculated value: " + f);
    }

    public static void test_batcher() throws FileNotFoundException {
        String path = "res/DPAv2_256/noisy_256class_1k_train.arff";
        Batcher batcher = new Batcher(new Parser(new Reader(path)), 3);
        // Build string by serializing batches.
        StringBuilder batches = new StringBuilder();
        Pair<float[][], float[]> d;
        while ((d = batcher.get()) != null)
            batches.append(new DatumF(d, null).toString(","));
        // Build string by serializing individual parses.
        StringBuilder parses = new StringBuilder();
        Parser parser = new Parser(new Reader(path));
        Pair<float[], Float> s;
        while ((s = parser.get()) != null)
            parses.append(new DatumF(s).toString(","));
        // Compare strings.
        System.out.println(batches.toString().substring(0, 1000));
        System.out.println(parses.toString().substring(0, 1000));
        System.out.println(batches.toString().equals(parses.toString()) ? "Both representations are same!" : "Representations differ!");
    }
}
