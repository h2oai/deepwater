package deepwater.backends.tensorflow.test;

import deepwater.backends.tensorflow.Client;
import org.junit.Test;

public class TestGRPC {

    @Test
    public void testSimpleConnection() throws InterruptedException {
            Client client = new Client("localhost", 50051);
            try {
                /* Access a service running on the local machine on port 50051 */
                client.Ping();
            } finally {
                client.shutdown();
            }
        }
    }
