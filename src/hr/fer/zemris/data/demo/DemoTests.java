package hr.fer.zemris.data.demo;

import hr.fer.zemris.data.Batcher;
import hr.fer.zemris.data.Modifier;
import hr.fer.zemris.data.Parser;
import hr.fer.zemris.data.Reader;
import hr.fer.zemris.data.modifiers.IModifier;
import hr.fer.zemris.data.primitives.BatchPair;
import hr.fer.zemris.data.primitives.DataPair;
import hr.fer.zemris.data.primitives.DatumF;
import hr.fer.zemris.utils.Stopwatch;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Arrays;


public class DemoTests {
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
        while (i++ < 5 && (s = reader.next()) != null) {
            System.out.println(s);
        }
    }

    public static void test_parser() throws FileNotFoundException {
        String path = "res/DPAv2_256/noisy_256class_1k_train.arff";
        Parser parser = new Parser(new Reader(path));
        int i = 0;
        DataPair s;
        while (i++ < 5 && (s = parser.next()) != null) {
            System.out.println(Arrays.toString(s.getKey()) + " " + s.getVal());
        }
        System.out.println();
        System.out.println(parser.describe());
    }

    @SuppressWarnings("Duplicates")
    public static void test_cacher() throws FileNotFoundException {
        String path = "res/noisy/30k/noisy_9class_30k.arff";
        IModifier[] modifiers = new IModifier[]{};
        Stopwatch stopwatch = new Stopwatch();
        DataPair p;

        // Test loading time and memory consumption.
        System.out.println("Testing loading...");
        long freeS, freeE;

        stopwatch.start();
        freeS = Runtime.getRuntime().freeMemory();
        Parser parser = new Parser(new Reader(path));
        freeE = Runtime.getRuntime().freeMemory();
        System.out.println("Loading parser: " + stopwatch.stop() + "ms\n                " + (freeS - freeE) + "B");

        stopwatch.start();
        freeS = Runtime.getRuntime().freeMemory();
        Modifier cacher = new Modifier(new Parser(new Reader(path)), modifiers);
        freeE = Runtime.getRuntime().freeMemory();
        System.out.println("Loading cacher: " + stopwatch.stop() + "ms\n                " + (freeS - freeE) + "B");

        // Test iteration time.
        System.out.println("Testing inference...");
        int repeats = 100;
        float f = 0;
        long t = 0;
        for (int i = 0; i < repeats; i++) {
            f = 0;
            stopwatch.start();
            while ((p = parser.next()) != null) f += p.getVal()[0];
            parser.reset();
            t += stopwatch.stop();
        }
        System.out.println("Iterating parser (avg): " + t / (double) repeats + "ms");
        System.out.println("Calculated test value: " + f);

        t = 0;
        for (int i = 0; i < repeats; i++) {
            f = 0;
            stopwatch.start();
            while ((p = cacher.next()) != null) f += p.getVal()[0];
            cacher.reset();
            t += stopwatch.stop();
        }
        System.out.println("Iterating cacher (avg): " + t / (double) repeats + "ms");
        System.out.println("Calculated test value: " + f);
    }

    public static void test_batcher() throws FileNotFoundException {
        String path = "res/DPAv2_256/noisy_256class_1k_train.arff";
        Batcher batcher = new Batcher(new Parser(new Reader(path)), 3);

        // Build string by serializing batches.
        StringBuilder batches = new StringBuilder();
        BatchPair d;
        while ((d = batcher.next()) != null)
            batches.append(new DatumF(d).toString(","));

        // Build string by serializing individual parses.
        StringBuilder parses = new StringBuilder();
        Parser parser = new Parser(new Reader(path));
        DataPair s;
        while ((s = parser.next()) != null)
            parses.append(new DatumF(s).toString(","));

        // Compare strings.
        System.out.println(batches.toString().substring(0, 1000));
        System.out.println(parses.toString().substring(0, 1000));
        System.out.println(batches.toString().equals(parses.toString()) ? "Both representations are same!" : "Representations differ!");
    }
}

