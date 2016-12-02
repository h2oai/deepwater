package deepwater.backends.tensorflow.test;

import deepwater.backends.tensorflow.Client;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.tensorflow.framework.MetaGraphDef;

public class TestGRPC {

    private Client client;

    @Before
    public void setUp(){
        client = new Client("localhost", 50051);
    }

    @After
    public void tearDown() throws InterruptedException {
        client.shutdown();
    }

    @Test
    public void testSimpleConnection() throws InterruptedException {
        client.Ping();
    }

    @Test
    public void testSimpleBuildNetwork() throws InterruptedException {
        MetaGraphDef graph = client.BuildNetwork("lenet");
    }

}
