package deepwater.backends.grpc;

import deepwater.backends.BackendModel;


public class XGRPCBackendModel implements BackendModel {
    private final byte[] model;
    private final byte[] uuid;
    private XGRPCBackendSession session;

    public XGRPCBackendModel(byte[] uuid, byte[] model) {
        this.model = model;
        this.uuid = uuid;

    }

    public void setSession(XGRPCBackendSession session) {
        this.session = session;
    }


    public XGRPCBackendSession getSession() {
        return session;
    }

    public byte[] getUUID() {
        return uuid;
    }

    public byte[] getState() {
        return model;
    }
}
