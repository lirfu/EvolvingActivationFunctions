package hr.fer.zemris.utils.threading;

import hr.fer.zemris.genetics.Utils;

public class WorkArbiter {
    private final String name_;
    private Worker[] workers_;

    public WorkArbiter(String name, int worker_number) {
        name_ = name;
        workers_ = new Worker[worker_number];
        for (int i = 0; i < worker_number; i++)
            workers_[i] = new Worker(name + '_' + i);
    }

    public void postWork(Work work) {
        // Selects the worker with the smallest queue.
        Utils.findLowest(workers_).enqueueWork(work);

        // Select workers in the defined order, distributing load uniformly.
//        workers_[mCurrentIndex].enqueueWork(work);
//        mCurrentIndex = ++mCurrentIndex % workers_.length;
    }

    public void kill() {
        for (Worker w : workers_)
            w.kill();
    }

    public void waitOn(WaitCondition condition) {
        while (!condition.satisfied()) {
            synchronized (this) {
                try {
                    wait(500);
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            }
        }
    }

    public String getName() {
        return name_;
    }

    public int getWorkerNumber() {
        return workers_.length;
    }

    @Override
    protected void finalize() throws Throwable {
        kill();
        super.finalize();
    }

    public interface WaitCondition {
        public boolean satisfied();
    }
}
