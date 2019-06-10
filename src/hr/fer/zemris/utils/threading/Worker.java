package hr.fer.zemris.utils.threading;

import java.util.LinkedList;

public class Worker {
    private boolean is_alive_ = true;
    private long wait_time_ = 500;
    private final String name_;
    private final Thread thread_;
    private final LinkedList<Work> queue_;
    private boolean idle_ = true;

    /**
     * Creates a new worker and starts the internal thread.
     */
    public Worker(String name, final LinkedList<Work> queue) {
        name_ = name;
        queue_ = queue;

        thread_ = new Thread(() -> {
            while (is_alive_) {
                // Deplete the queue.
                while (queue_.size() > 0) {
                    Work w;
                    synchronized (queue_) {
                        if (queue_.size() == 0) continue;
                        w = queue_.removeFirst();
                        idle_ = false;
                    }
                    w.work();
                }
                idle_ = true;

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
        thread_.setName(name);
        thread_.setDaemon(true);
        thread_.start();
    }

    public boolean isIdle() {
        return idle_;
    }

    /**
     * Returns workers' name.
     */
    public final String getName() {
        return name_;
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
}
