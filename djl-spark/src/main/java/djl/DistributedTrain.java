package djl;

import ai.djl.MalformedModelException;
import ai.djl.Model;
import ai.djl.metric.Metrics;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.nn.Block;
import ai.djl.nn.Parameter;
import ai.djl.nn.ParameterList;
import ai.djl.training.GradientCollector;
import ai.djl.training.Trainer;
import ai.djl.training.dataset.Batch;
import ai.djl.training.dataset.Dataset;
import ai.djl.training.listener.TrainingListener;
import ai.djl.translate.TranslateException;
import ai.djl.util.Pair;
import ai.djl.util.Preconditions;
import djl.server.Server;
import djl.training.DistributedTrainer;
import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.sql.SparkSession;

import java.io.*;
import java.util.Iterator;
import java.util.*;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.ReentrantLock;
import java.util.stream.Collectors;


@SuppressWarnings({"rawtypes", "unchecked"})
public class DistributedTrain {

    private final static int PORT = 1888;

    private DistributedTrain() {
    }

    public static void fit(Trainer trainer, int numEpoch, Dataset trainingDataset, Dataset validateDataset, SparkSession spark, int partitionNum) throws IOException, TranslateException, InterruptedException, MalformedModelException {
        JavaSparkContext javaSparkContext = JavaSparkContext.fromSparkContext(spark.sparkContext());

        JavaRDD<Pair<byte[], byte[]>> data = parallelizeDataset(trainer, trainingDataset, javaSparkContext);
        Broadcast<Map<String, byte[]>> broadcastParameters = javaSparkContext.broadcast(encodeParameters(trainer.getModel().getBlock().getParameters()));
        distributedTrain(trainer, data, numEpoch, partitionNum, broadcastParameters);

        evaluateDataset(trainer, validateDataset);
    }

    public static void distributedTrain(Trainer trainer, JavaRDD<Pair<byte[], byte[]>> data, int numEpoch,int partitionNum, Broadcast<Map<String, byte[]>> broadcastParameters) throws InterruptedException, IOException, MalformedModelException {
        AtomicInteger parameterNum = new AtomicInteger();
        trainer.getModel().getBlock().getParameters().forEach(
                (parameterPair) -> {
                    if (parameterPair.getValue().requiresGradient()) {
                        parameterNum.getAndIncrement();
                    }
                }
        );

        ReentrantLock serverStartLock = new ReentrantLock();
        Thread serverThread = new Thread(new Runnable() {
            @Override
            public void run() {
                Server server = new Server(PORT);
                server.start(partitionNum, parameterNum.get(), serverStartLock, trainer);
            }
        });
        serverThread.setName("serverThread");
        serverThread.start();

        synchronized (serverStartLock) {
            serverStartLock.wait();
            System.out.println("Start training!");
            List<byte[]> result = data.mapPartitions(new FlatMapFunction() {

                @Override
                public Iterator call(Object o) throws Exception {
                    Block block = Run.getBlock();

                    ParameterList parameterList = block.getParameters();
                    for(int i = 0; i < parameterList.size(); i++){
                        parameterList.get(i).getValue().setId(String.valueOf(i+1));
                    }
                    Model distributedModel = Model.newInstance("distributedModel");

                    distributedModel.setBlock(block);

                    DistributedTrainer distributedTrainer = new DistributedTrainer(distributedModel, Run.setupTrainingConfig(null), "0.0.0.0", PORT);
                    distributedTrainer.initialize(Run.getShape());
                    distributedTrainer.setMetrics(new Metrics());

                    Iterator<Pair<byte[], byte[]>> dataList = (Iterator<Pair<byte[], byte[]>>) o;

                    Pair<byte[], byte[]> dataLabelPair = dataList.next();
                    NDList data = NDList.decode(distributedTrainer.getManager(), dataLabelPair.getKey());
                    NDList label = NDList.decode(distributedTrainer.getManager(), dataLabelPair.getValue());

                    int partitionLabel = Integer.parseInt(data.get(0).getName());
                    for (int i = 0; i < numEpoch; i++) {
                        try(GradientCollector gradientCollector = distributedTrainer.newGradientCollector()){
                            NDList pred = distributedTrainer.forward(data, label);
                            NDArray loss = distributedTrainer.getLoss().evaluate(label, pred);
                            gradientCollector.backward(loss);
                            distributedTrainer.step();

                            System.out.println("finished epoch " + String.valueOf(i+1));
                        }
                    }



                    List<byte[]> resultParameters = new LinkedList<>();
                    for(int i = 0 ; i < parameterList.size(); i++){
                        if(i % partitionNum == partitionLabel){
                            ByteArrayOutputStream bos = new ByteArrayOutputStream();
                            DataOutputStream dos = new DataOutputStream(bos);
                            dos.write(i);
                            parameterList.get(i).getValue().save(dos);

                            resultParameters.add(bos.toByteArray());
                        }
                    }

                    distributedModel.close();
                    distributedTrainer.close();
                    return resultParameters.iterator();
                }
            }, true).collect();

            ParameterList parameterList = trainer.getModel().getBlock().getParameters();
            for(int i = 0; i < result.size(); i++){
                byte[] bytes= result.get(i);

                ByteArrayInputStream bis = new ByteArrayInputStream(bytes);
                DataInputStream dis = new DataInputStream(bis);
                int number = dis.read();

                parameterList.get(number).getValue().load(trainer.getManager(), dis);
            }
        }
    }

    private static JavaRDD<Pair<byte[], byte[]>> parallelizeDataset(Trainer trainer, Dataset dataset, JavaSparkContext spark) throws IOException, TranslateException {
        Iterator iterator = trainer.iterateDataset(dataset).iterator();

        List<Pair<byte[], byte[]>> batchList = new ArrayList<>();
        int counter = 0;

        while(iterator.hasNext()){
            Batch batch = (Batch)iterator.next();

            batch.getData().get(0).setName(String.valueOf(counter++));
            batchList.add(new Pair(batch.getData().encode(), batch.getLabels().encode()));
        }

        return spark.parallelize(batchList);
    }

    private static Map<String, byte[]> encodeParameters(ParameterList parameterList) throws IOException {
        Map<String, byte[]> result = new HashMap();

        for(Pair<String, Parameter> parameters: parameterList){
            ByteArrayOutputStream bos = new ByteArrayOutputStream();
            DataOutputStream dos = new DataOutputStream(bos);
            parameters.getValue().save(dos);

            result.put(parameters.getValue().getId(),bos.toByteArray());
        }

        return result;
    }

    /**
     * Evaluates the test dataset.
     *
     * @param trainer the trainer to evaluate on
     * @param testDataset the test dataset to evaluate
     * @throws IOException for various exceptions depending on the dataset
     * @throws TranslateException if there is an error while processing input
     */
    public static void evaluateDataset(Trainer trainer, Dataset testDataset)
            throws IOException, TranslateException {

        if (testDataset != null) {
            for (Batch batch : trainer.iterateDataset(testDataset)) {
                validateBatch(trainer, batch);
                batch.close();
            }
        }
    }
    /**
     * Validates the given batch of data.
     *
     * <p>During validation, the evaluators and losses are computed, but gradients aren't computed,
     * and parameters aren't updated.
     *
     * @param trainer the trainer to validate the batch with
     * @param batch a {@link Batch} of data
     * @throws IllegalArgumentException if the batch engine does not match the trainer engine
     */
    public static void validateBatch(Trainer trainer, Batch batch) {
        Preconditions.checkArgument(
                trainer.getManager().getEngine() == batch.getManager().getEngine(),
                "The data must be on the same engine as the trainer. You may need to change one of your NDManagers.");
        Batch[] splits = batch.split(trainer.getDevices(), false);
        TrainingListener.BatchData batchData =
                new TrainingListener.BatchData(batch, new ConcurrentHashMap<>(), new ConcurrentHashMap<>());

        if (splits.length > 1 && trainer.getExecutorService().isPresent()) {
            // multi-threaded
            ExecutorService executor = trainer.getExecutorService().get();
            List<CompletableFuture<Boolean>> futures = new ArrayList<>(splits.length);
            for (Batch split : splits) {
                futures.add(
                        CompletableFuture.supplyAsync(
                                () -> validateSplit(trainer, batchData, split), executor));
            }
            CompletableFuture.allOf(futures.stream().toArray(CompletableFuture[]::new));
        } else {
            // sequence
            for (Batch split : splits) {
                validateSplit(trainer, batchData, split);
            }
        }

        trainer.notifyListeners(listener -> listener.onValidationBatch(trainer, batchData));
    }

    private static boolean validateSplit(Trainer trainer, TrainingListener.BatchData batchData, Batch split) {
        NDList data = split.getData();
        NDList labels = split.getLabels();
        NDList preds = trainer.evaluate(data);
        batchData.getLabels().put(labels.get(0).getDevice(), labels);
        batchData.getPredictions().put(preds.get(0).getDevice(), preds);
        return true;
    }
}
