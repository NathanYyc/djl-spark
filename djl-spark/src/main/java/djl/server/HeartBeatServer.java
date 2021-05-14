package djl.server;

import ai.djl.training.Trainer;
import djl.Util.Grads;
import djl.initializer.HeartBeatServerInitializer;
import djl.initializer.ParameterServerInitializer;
import io.netty.bootstrap.ServerBootstrap;
import io.netty.channel.ChannelFuture;
import io.netty.channel.EventLoopGroup;
import io.netty.channel.nio.NioEventLoopGroup;
import io.netty.channel.socket.nio.NioServerSocketChannel;
import io.netty.handler.logging.LogLevel;
import io.netty.handler.logging.LoggingHandler;
import io.netty.util.ResourceLeakDetector;
import org.apache.hadoop.yarn.webapp.hamlet.Hamlet;

import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.ReentrantLock;

public class HeartBeatServer {
    private int port;
    private EventLoopGroup bossGroup;
    private EventLoopGroup workerGroup;
    private ChannelFuture channelFuture;

    private AtomicInteger connectionCount;

    public HeartBeatServer(int port){
        this.port = port;
        connectionCount = new AtomicInteger(0);
    }

    public void start(Object lock){
        System.out.println("Heart Beat HeartBeatServer started!");
        bossGroup = new NioEventLoopGroup();
        workerGroup = new NioEventLoopGroup();
        try{

            synchronized (lock) {
                ServerBootstrap serverBootstrap = new ServerBootstrap();
                serverBootstrap.group(bossGroup, workerGroup)
                        .handler(new LoggingHandler(LogLevel.INFO))
                        .channel(NioServerSocketChannel.class)
                        .childHandler(new HeartBeatServerInitializer(connectionCount));
                channelFuture = serverBootstrap.bind(port).sync();

                lock.notify();
            }
            channelFuture.channel().closeFuture().sync();

            ResourceLeakDetector.setLevel(ResourceLeakDetector.Level.ADVANCED);

        } catch (InterruptedException e) {
            e.printStackTrace();
        } finally{
            bossGroup.shutdownGracefully();
            workerGroup.shutdownGracefully();
        }
    }


}
