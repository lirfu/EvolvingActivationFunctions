package hr.fer.zemris.utils.threading;

import java.util.LinkedList;

public class WorkArbiter {
    private final String name_;
    private Worker[] workers_;
    private final LinkedList<Work> queue_;

    public WorkArbiter(String name, int worker_number) {
        name_ = name;
        queue_ = new LinkedList<>();
        workers_ = new Worker[worker_number];
        for (int i = 0; i < worker_number; i++)
            workers_[i] = new Worker(name + '_' + i, queue_);
    }

    public void postWork(Work work) {
        // Add to joint queue.
        synchronized (queue_) {
            queue_.addLast(work);
        }
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
     * Returns a condition that is satisfied when the queue is empty and all workers are done working.
     */
    public WaitCondition getAllFinishedCondition() {
        return () -> {
            if (queue_.size() > 0) return false;
            for (Worker w : workers_)
                if (!w.isIdle())
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
        sb.append("Work to do: ").append(queue_.size());
        sb.append("\nWorkers: ");
        int i = 0;
        for (Worker w : workers_) {
            if (i > 0) sb.append(", ");
            sb.append(w.getName());
            i++;
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
