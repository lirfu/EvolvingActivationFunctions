package hr.fer.zemris.utils.threading;

import org.jetbrains.annotations.NotNull;

import java.util.LinkedList;

public class Worker implements Comparable<Worker> {
    private boolean is_alive_ = true;
    private long wait_time_ = 500;
    private final String ID_;
    private final Thread thread_;
    private final LinkedList<Work> queue_;

    /**
     * Creates a new worker and starts the internal thread.
     */
    public Worker(String id) {
        ID_ = id;
        queue_ = new LinkedList<>();

        thread_ = new Thread(() -> {
            while (is_alive_) {
                // Deplete the queue.
                while (queue_.size() > 0) {
                    Work w;
                    synchronized (Worker.this) {
                        w = queue_.pop();
                    }
                    w.work();
                }

                // Wait for changes in queue or timeout.
                synchronized (Worker.this) {
                    try {
                        wait(wait_time_);
                    } catch (InterruptedException e) {
                        e.printStackTrace();
                    }
                }
            }
        });
        thread_.setDaemon(true);
        thread_.start();
    }

    /**
     * Enqueues work to workers' queue. If worker is dead, work is dropped.
     */
    public final synchronized void enqueueWork(Work work) {
        if (!is_alive_) return;
        queue_.push(work);
        notify();
    }

    /**
     * Returns the current queue size.
     */
    public final int getWorkCount() {
        return queue_.size();
    }

    /**
     * Returns workers' name.
     */
    public final String getID() {
        return ID_;
    }

    /**
     * Sets the wait time when queue gets emptied.
     */
    public void setWaitTime(long wait_time) {
        wait_time_ = wait_time;
    }

    /**
     * Kills the worker and notifies his listeners. May he rest in peace for all of eternity.
     */
    public final void kill() {
        is_alive_ = false;
        notify();
    }

    @Override
    public final int compareTo(@NotNull Worker o) {
        return getWorkCount() - o.getWorkCount();
    }
}
