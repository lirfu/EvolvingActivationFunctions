package hr.fer.zemris.utils.threading;

import org.junit.Test;

import java.util.Random;

public class WorkArbiterTest {
    private WorkArbiter arbiter_ = new WorkArbiter("Arbi", 3);

    @Test
    public void test() {
        final Random r = new Random();
        final int[] i = new int[1];
        Work w = () -> {
            synchronized (this) {
                try {
                    wait(r.nextInt(10) + 1);
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            }
            synchronized (i) {
                System.out.println(" " + i[0]);
                i[0]++;
            }
        };

        int T = 10 / 2;

        // Generate rapidly.
        for (int t = 0; t < T; t++)
            arbiter_.postWork(w);

        // Wait a little.
        synchronized (this) {
            try {
                wait(300);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }

        // Generate with pauses.
        for (int t = 0; t < T; t++) {
            arbiter_.postWork(w);
            synchronized (this) {
                try {
                    wait(500);
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            }
        }

        // Wait until all finish.
        synchronized (this) {
            try {
                while (i[0] < 2 * T) {
                    wait(500);
                }
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }

        System.out.println("Done!");
    }
}