package experimenting_stuff;

import hr.fer.zemris.utils.threading.Work;
import hr.fer.zemris.utils.threading.WorkArbiter;

public class TestStuff {
    @org.junit.Test
    public void workerIdentification() {
        WorkArbiter a = new WorkArbiter("A", 3);
        Work w = () -> System.out.println(Thread.currentThread().getName());
        for (int i = 0; i < 10; i++)
            a.postWork(w);
    }
}
