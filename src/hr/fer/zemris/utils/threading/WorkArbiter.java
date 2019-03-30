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

    /**
     * Waits until the condition is satisfied (until it is true).
     */
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

    /**
     * Returns a condition that is satisfied when all the worker queues are empty.
     * Use with caution, worker queues can become empty by chance or they can constantly get filled making an infinite loop.
     * When unsure, construct your own wait condition.
     */
    public WaitCondition getFinishedCondition() {
        return () -> {
            for (Worker w : workers_)
                if (w.getWorkCount() > 0)
                    return false;
            return true;
        };
    }

    public String getName() {
        return name_;
    }

    public int getWorkerNumber() {
        return workers_.length;
    }

    public String getStatus() {
        StringBuilder sb = new StringBuilder();
        for (Worker w : workers_) {
            sb.append(w.getName()).append(": ").append(w.getWorkCount()).append('\n');
        }
        return sb.toString();
    }

    @Override
    protected void finalize() throws Throwable {
        kill();
        super.finalize();
    }

    public static String getCurrentWorkerName() {
        return Thread.currentThread().getName();
    }

    public interface WaitCondition {
        public boolean satisfied();
    }
}
