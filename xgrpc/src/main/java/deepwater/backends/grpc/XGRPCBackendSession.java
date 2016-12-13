package deepwater.backends.grpc;

import deepwater.backends.BackendModel;

public class XGRPCBackendSession implements BackendModel {

    private final String handle;

    public XGRPCBackendSession(String handle) {
        this.handle = handle;
    }

    public String getHandle() {
        return handle;
    }
}
