package hr.fer.zemris.utils.threading;

public class WorkArbiter {
    private final String mName;
    private Worker[] mWorkers;
    private int mCurrentIndex = 0;

    public WorkArbiter(String name, int workerNumber) {
        mName = name;
        mWorkers = new Worker[workerNumber];
        for (int i = 0; i < workerNumber; i++)
            mWorkers[i] = new Worker(name + '_' + i);
    }

    public void postWork(Work work) {
        mWorkers[mCurrentIndex].enqueueWork(work);
        mCurrentIndex = ++mCurrentIndex % mWorkers.length;
    }

    public void kill() {
        for (Worker w : mWorkers)
            w.kill();
    }

    public String getName() {
        return mName;
    }

    public int getWorkerNumber() {
        return mWorkers.length;
    }
}
