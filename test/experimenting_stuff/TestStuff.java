package experimenting_stuff;

import hr.fer.zemris.genetics.symboregression.IExecutable;
import hr.fer.zemris.utils.Counter;
import hr.fer.zemris.utils.threading.Work;
import hr.fer.zemris.utils.threading.WorkArbiter;
import org.junit.Test;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Random;

public class TestStuff {
    @Test
    public void workerIdentification() {
        WorkArbiter a = new WorkArbiter("A", 3);
        Work w = () -> System.out.println(Thread.currentThread().getName());
        for (int i = 0; i < 10; i++)
            a.postWork(w);
    }

    @Test
    public void testExecutableParallelism(){
        // Test if an Executor with local variables run in parallel mixes those variables between threads.
        // In other words, I worry that multiple threads running the same Executor will use the same
        // references of local variables, producing a very different result.

        Nd4j.setDataType(DataBuffer.Type.FLOAT);
        WorkArbiter arbiter = new WorkArbiter("test", 10);

        final IExecutable<Counter, Counter> e = (input, node) -> {
            INDArray arr = Nd4j.scalar(100f);
            float val = input.value();
            input.increment();
            arr.addi(input.value());
            if ((int) (arr.getFloat(0)) != (int) (100 + val + 1))
                System.err.println("ERROR! " + ((int) arr.getFloat(0)) + " != " + (int) (100 + val + 1));
//            else
//                System.out.println("ok");
            return input;
        };

        final Counter c = new Counter();

        Work w = () -> {
            int start = new Random().nextInt(100);
            Counter ctr = new Counter(start);
//            System.out.println(Thread.currentThread().getName() + " starts with: " + start);
            for (int i = 0; i < 1000; i++) {
                e.execute(ctr, null);
            }
            int end = ctr.value();
            if (end != start + 1000)
                System.err.println(Thread.currentThread().getName() + " ends wrong!");
            synchronized (c) {
                System.out.println("Finished " + c.increment().value() + "/1000");
            }
        };

        for (int i = 0; i < 1000; i++) {
            arbiter.postWork(w);
        }

        arbiter.waitOn(arbiter.getAllFinishedCondition());

        System.out.println("DONE!");
    }
}
