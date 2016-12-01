package deepwater.backends.tensorflow;

import ai.h2o.deepwater.ServiceGrpc;
import ai.h2o.deepwater.PingRequest;
import ai.h2o.deepwater.Status;
import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import java.util.concurrent.TimeUnit;
import java.util.logging.Logger;


public class Client {
    private static final Logger logger = Logger.getLogger(Client.class.getName());

    private final ServiceGrpc.ServiceBlockingStub blockingStub;
    private final ManagedChannel channel;

    /** Construct client connecting to HelloWorld server at {@code host:port}. */
    public Client(String host, int port) {
        this(ManagedChannelBuilder.forAddress(host, port)
                // Channels are secure by default (via SSL/TLS). For the example we disable TLS to avoid
                // needing certificates.
                .usePlaintext(true));
    }

    Client(ManagedChannelBuilder<?> channelBuilder) {
        channel = channelBuilder.build();
        blockingStub = ServiceGrpc.newBlockingStub(channel);
    }

    public void shutdown() throws InterruptedException {
        channel.shutdown().awaitTermination(5, TimeUnit.SECONDS);
    }

    public void Ping() {
        PingRequest request = PingRequest.newBuilder().build();
        Status response;
        response = blockingStub.ping(request);
        logger.info("Ping: " + response);
    }

}
