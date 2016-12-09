package deepwater.backends.grpc;

import com.google.protobuf.ByteString;
import deepwater.backends.BackendModel;

public class XGRPCBackendSession implements BackendModel {
    private final ByteString state;

    public XGRPCBackendSession(ByteString network) {
        this.state = network;
    }

    public ByteString getState() {
        return state;
    }
}
