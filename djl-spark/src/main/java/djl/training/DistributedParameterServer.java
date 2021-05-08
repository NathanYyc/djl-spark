package djl.training;

import ai.djl.ndarray.NDArray;
import ai.djl.nn.Parameter;
import ai.djl.training.ParameterServer;
import djl.server.Client;
import io.netty.buffer.ByteBuf;
import io.netty.buffer.Unpooled;


public class DistributedParameterServer implements ParameterServer {


    private Client client;
    private ParameterServer nativeParameterServer;

    public DistributedParameterServer(int serverPort, ParameterServer parameterServer) throws InterruptedException {
        nativeParameterServer = parameterServer;
        this.client = new Client(serverPort);
    }
    /**
     * Initializes the {@code ParameterStore} for the given parameter.
     *
     * @param parameterId the parameter ID
     * @param value       the values to be set for the given parameter
     */
    public void init(String parameterId, NDArray[] value) {
        nativeParameterServer.init(parameterId, value);
    }

    /**
     * Updates the parameter of a key from Parameter Server.
     *
     * @param parameterId the key to identify the parameter
     * @param grads       the gradient NDArrays in different devices to apply the update.
     * @param params      the parameter NDArrays in different devices to be updated.
     */
    public void update(String parameterId, NDArray[] grads, NDArray[] params) {
        //get the sum of grads from all the devices
        NDArray sum = grads[0].duplicate();
        sum.setName(params[0].getName());

        for(int i = 1; i< grads.length; i++){
            sum.add(grads[i]);
        }

        //deliver encoded sum to the ps
        byte[] listByte = sum.encode();
        ByteBuf input = Unpooled.wrappedBuffer(listByte);
        ByteBuf result = null;

        try {
            result = client.getResult(input);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }finally{
            sum.detach();
            sum.close();
        }


        //get sum from ps & update all the grads
        byte[] resultArray = new byte[result.nioBuffer().remaining()];
        result.readBytes(resultArray);
        sum = NDArray.decode(grads[0].getManager(), resultArray);


        //update the grads
        for(int i = 0; i < grads.length; i++){
            NDArray temp = null;
            try{
                temp = grads[i];
                grads[i] = sum.duplicate().toDevice(grads[i].getDevice(), true);
            }finally{
                if (temp != null) {
                    temp.detach();
                    temp.close();
                }
            }
        }
        nativeParameterServer.update(parameterId, grads, params);

        sum.detach();
        sum.close();
        result.release();
    }

    /**
     * {@inheritDoc}
     */
    public void close() {
        nativeParameterServer.close();
        try {
            client.close();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }
}
