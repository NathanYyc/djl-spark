package djl.Util;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import djl.server.Client;
import io.netty.buffer.ByteBuf;
import io.netty.buffer.Unpooled;
import io.netty.channel.ChannelFuture;
import io.netty.channel.ChannelFutureListener;

import java.util.Iterator;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.CountDownLatch;

public class Dispatcher {
    private static Set<String> serverIPs;
    private static Map<String, Client> connections;

    public static void initial(Set<String> serverIPs){
        Dispatcher.serverIPs = serverIPs;
    }

    public static NDArray dispatch(NDArray ndArray, NDManager ndManager) throws InterruptedException{
        int length = serverIPs.size();

        NDList list = ndArray.flatten().split(length);

        Iterator<String> iterator = serverIPs.iterator();
        NDList list1 = new NDList();
        int index = 0;
        NDArray[] resultArrays = new NDArray[length];
        CountDownLatch countDownLatch = new CountDownLatch(length);

        while(iterator.hasNext()){
            int finalIndex = index;
            new Thread(new Runnable() {
                @Override
                public void run() {
                    try {
                        ByteBuf result =connections.get(iterator.next()).getResult(Unpooled.wrappedBuffer(list.get(finalIndex).encode()));
                        byte[] resultArray = new byte[result.nioBuffer().remaining()];
                        result.readBytes(resultArray);
                        NDArray sum = NDArray.decode(ndManager, resultArray);

                        resultArrays[finalIndex] = sum;

                        countDownLatch.countDown();
                    } catch (InterruptedException e) {
                        e.printStackTrace();
                    }
                }
            }).start();

            index++;
        }

        countDownLatch.await();
        for(int i = 1; i < length ;i++){
            resultArrays[0].concat(resultArrays[i]);

            resultArrays[i].detach();
            resultArrays[i].close();
        }

        return resultArrays[0];
    }
}
