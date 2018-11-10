package hr.fer.zemris.utils.threading;

public class WorkArbiter {
    private Worker[] mWorkers;
    private int mCurrentIndex = 0;

    public WorkArbiter(int workerNumber) {
        mWorkers = new Worker[workerNumber];
        for (int i = 0; i < workerNumber; i++)
            mWorkers[i] = new Worker(i);
    }

    public void postWork(Work work) {
        mWorkers[mCurrentIndex].enqueueWork(work);
        mCurrentIndex = ++mCurrentIndex % mWorkers.length;
    }

    public void kill() {
        for (Worker w : mWorkers)
            w.kill();
    }

    public int getWorkerNumber() {
        return mWorkers.length;
    }
}
