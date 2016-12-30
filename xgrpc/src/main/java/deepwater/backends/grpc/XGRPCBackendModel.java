package deepwater.backends.grpc;

import deepwater.backends.BackendModel;
import deepwater.datasets.ImageDataSet;


public class XGRPCBackendModel implements BackendModel {
    private final byte[] model;
    private final byte[] uuid;
    private XGRPCBackendSession session;
    private ImageDataSet dataset;

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

// FIXME: this should not be part of the implementation
    public void setDataset(ImageDataSet dataset) {
        this.dataset = dataset;
    }

    public ImageDataSet getBackend() {
        return this.dataset;
    }
}
