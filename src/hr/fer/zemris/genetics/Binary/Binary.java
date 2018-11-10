package hr.fer.zemris.genetics.Binary;

import hr.fer.zemris.genetics.Genotype;

import java.util.Random;

public class Binary extends Genotype<Boolean> {
    private int mParamNum;
    private double mMin;
    private double mMax;
    private double mPrecision;
    private boolean mUseGray;

    private int mBitsPerParam;
    private boolean[][] mBits;

    public Binary(int paramNum, double min, double max, int precision, boolean useGray) {
        super();
        mParamNum = paramNum;
        mMin = min;
        mMax = max;
        mPrecision = precision;
        mUseGray = useGray;

        mBitsPerParam = (int) Math.ceil(Math.log((max - min) * Math.pow(10, precision) + 1) / Math.log(2));
        mBits = new boolean[paramNum][mBitsPerParam];
    }

    public Binary(Binary b) {
        super(b);
        mParamNum = b.mParamNum;
        mMin = b.mMin;
        mMax = b.mMax;
        mPrecision = b.mPrecision;
        mUseGray = b.mUseGray;

        mBitsPerParam = b.mBitsPerParam;
        mBits = new boolean[mParamNum][mBitsPerParam];
        for (int i = 0; i < mParamNum; i++)
            System.arraycopy(b.mBits[i], 0, mBits[i], 0, mBits[i].length);
    }

    @Override
    public Boolean get(int index) {
        int param = index / mBitsPerParam;
        int second = index - param * mBitsPerParam;
        return mBits[param][second];
    }

    @Override
    public void set(int index, Boolean value) {
        int param = index / mBitsPerParam;
        int second = index - param * mBitsPerParam;
        mBits[param][second] = value;
    }

    @Override
    public int size() {
        return mParamNum * mBitsPerParam;
    }

    @Override
    public Genotype copy() {
        return new Binary(this);
    }

    @Override
    public void randomize(Random rand) {
        for (int i = 0; i < mParamNum; i++)
            for (int j = 0; j < mBitsPerParam; j++)
                mBits[i][j] = rand.nextBoolean();
    }

    @Override
    public String stringify() {
        StringBuilder str = new StringBuilder();
        for (int i = 0; i < mParamNum; i++) {
            str.append('(');
            for (int j = 0; j < mBitsPerParam; j++)
                str.append(mBits[i][j] ? 1 : 0);
            str.append(')');
        }
        str.append('\n');
        str.append('(');
        double[] vals = decode();
        for (int i = 0; i < mParamNum; i++) {
            if (i > 0) str.append(", ");
            str.append(vals[i]);
        }
        str.append(')');
        return str.toString();
    }

    @Override
    public Boolean generateParameter(Random rand) {
        return rand.nextBoolean();
    }

    public double[] decode() {
        double[] vals = new double[mParamNum];

        for (int i = 0; i < vals.length; i++) {
            int b = 0;
            int pot = 1;
            for (int j = mBits[i].length - 1; j >= 0; j--) {
                if (mBits[i][j])
                    b += pot;
                pot *= 2;
            }

            if (mUseGray) {
                b = b ^ (b >> 1);
            }

            vals[i] = mMin + (mMax - mMin) * b / (Math.pow(2, mBitsPerParam) - 1);
        }

        return vals;
    }
}
