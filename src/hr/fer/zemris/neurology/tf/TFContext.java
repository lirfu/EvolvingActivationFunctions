package hr.fer.zemris.neurology.tf;

import org.tensorflow.Operand;
import org.tensorflow.Session;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class TFContext {
    public TFContext addTarget(Operand<?> op) {
        targetsToDiscard.add(op.asOutput());
        return this;
    }

    public TFContext addTargetToFetch(Operand<?> op) {
        targetsToFetch.add(op);
        return this;
    }

    public TFStep createStep(Session session) {
        return new TFStep(session, this);
    }

    private final List<Operand<?>> targetsToDiscard = new ArrayList<>();
    private final List<Operand<?>> targetsToFetch = new ArrayList<>();

    List<Operand<?>> targetsToDiscard() {
        return Collections.unmodifiableList(targetsToDiscard);
    }

    List<Operand<?>> targetsToFetch() {
        return Collections.unmodifiableList(targetsToFetch);
    }
}
