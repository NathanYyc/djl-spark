package djl.training;

import ai.djl.ndarray.NDArray;
import djl.Util.Dispatcher;
import djl.server.Client;

import java.util.concurrent.BrokenBarrierException;


public class DistributedParameterServer implements ai.djl.training.ParameterServer {


    private Client client;
    private ai.djl.training.ParameterServer nativeParameterParameterServer;

    public DistributedParameterServer(int serverPort, ai.djl.training.ParameterServer parameterServer) throws InterruptedException {
        nativeParameterParameterServer = parameterServer;
        this.client = new Client(serverPort);
    }
    /**
     * Initializes the {@code ParameterStore} for the given parameter.····
     *
     * @param parameterId the parameter ID
     * @param value       the values to be set for the given parameter
     */
    public void init(String parameterId, NDArray[] value) {
        nativeParameterParameterServer.init(parameterId, value);
    }

    /**
     * Updates the parameter of a key from Parameter ParameterServer.
     *
     * @param parameterId the key to identify the parameter
     * @param grads       the gradient NDArrays in different devices to apply the update.
     * @param params      the parameter NDArrays in different devices to be updated.
     */
    public void update(String parameterId, NDArray[] grads, NDArray[] params){
        //get the sum of grads from all the devices
        NDArray sum = grads[0].duplicate();
        sum.setName(params[0].getName());

        for(int i = 1; i< grads.length; i++){
            sum.add(grads[i]);
        }
        sum.div(grads.length);

//        //deliver encoded sum to the ps
//        byte[] listByte = sum.encode();
//        ByteBuf input = Unpooled.wrappedBuffer(listByte);
//        ByteBuf result = null;
//
//        try {
//            result = client.getResult(input);
//        } catch (InterruptedException e) {
//            e.printStackTrace();
//        }finally{
//            sum.detach();
//            sum.close();
//        }
//
//
//        //get sum from ps & update all the grads
//        byte[] resultArray = new byte[result.nioBuffer().remaining()];
//        result.readBytes(resultArray);
//        sum = NDArray.decode(grads[0].getManager(), resultArray);
//
//
//        //update the grads
//        for(int i = 0; i < grads.length; i++){
//            NDArray temp = null;
//            try{
//                temp = grads[i];
//                grads[i] = sum.duplicate().toDevice(grads[i].getDevice(), true);
//            }finally{
//                if (temp != null) {
//                    temp.detach();
//                    temp.close();
//                }
//            }
//        }
        NDArray result = null;
        try {
            result = Dispatcher.dispatch(sum, grads[0].getManager());
        } catch (InterruptedException e) {
            e.printStackTrace();
        }

        for(int i = 0 ; i < grads.length; i++){
            grads[i] = result.duplicate().toDevice(grads[i].getDevice(), true);
        }

        nativeParameterParameterServer.update(parameterId, grads, params);

        sum.detach();
        sum.close();

        result.detach();
        result.close();
    }

    /**
     * {@inheritDoc}
     */
    public void close() {
        nativeParameterParameterServer.close();
        try {
            client.close();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }
}
