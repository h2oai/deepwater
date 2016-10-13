package deepwater.datasets;

import java.util.AbstractMap.SimpleImmutableEntry;

public class Pair<F, S> extends SimpleImmutableEntry<F, S> {

    public  Pair( F f, S s ) {
        super( f, s );
    }

    public F getFirst() {
        return getKey();
    }

    public S getSecond() {
        return getValue();
    }

    public String toString() {
        return "["+getKey()+","+getValue()+"]";
    }

}