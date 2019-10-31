/*
 * Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance
 * with the License. A copy of the License is located at
 *
 * http://aws.amazon.com/apache2.0/
 *
 * or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
 * OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions
 * and limitations under the License.
 */
package ai.djl.nn;

import ai.djl.MalformedModelException;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.LayoutType;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.ParameterStore;
import ai.djl.training.initializer.Initializer;
import ai.djl.util.PairList;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.List;

/**
 * A {@code Block} is a composable unit that forms a neural network.
 *
 * <p>Primarily, a {@code Block} is a unit that can take multiple inputs in the form of an {@link
 * NDList}, perform an operation on the inputs, and returns the output of the operation as an {@link
 * NDList}. In that sense, a {@code Block} is a neuron in the neural network. It can also be a
 * sub-network in a network, or the whole network itself. To understand the {@code Block}, we need
 * to understand the important components of a block:
 *
 * <ul>
 *   <li>Forward function
 *   <li>Parameters
 *   <li>Child blocks
 * </ul>
 *
 * <p>The core purpose of a {@code Block} is to perform an operation on the inputs, and return an
 * output. It is defined in the {@link #forward(ParameterStore, NDList) forward} method. The forward
 * function could be defined explicitly in terms of parameters or implicitly and could be a
 * combination of the functions of the child blocks.
 *
 * <p>The parameters of a {@code Block} are instances of {@link Parameter} which are required for
 * the operation in the forward function. For example, in a {@link ai.djl.nn.convolutional.Conv2D}
 * block, the parameters are {@code weight} and {@code bias}. During training, these parameters are
 * updated to reflect the training data, and that forms the crux of learning.
 *
 * <p>A block and its children together form a neural network. The way the network is structured
 * depends on the implementation of the block, and its interaction with its children. The children
 * can be aligned in sequence (eg. {@link SequentialBlock}) or in parallel (eg. {@link
 * ParallelBlock}), or in many other ways.
 *
 * <p>A block does not necessarily have to children and parameters. For example, {@link
 * SequentialBlock}, and {@link ParallelBlock} don't have any parameters, but do have child blocks.
 * Similarly, {@link ai.djl.nn.convolutional.Conv2D} does not have children but has parameters. It
 * is recommended for blocks that don't have children to extend {@link ParameterBlock}. There can be
 * special cases where blocks have neither parameters nor children. One such example is {@link
 * LambdaBlock}. {@link LambdaBlock} takes in a function, and applies that function to its input in
 * the {@link #forward(ParameterStore, NDList) forward} method.
 *
 * <p>Now that we understand the components of the block, we can explore what the block really
 * represents. A block combined with the recursive, hierarchical structure of its children together
 * forms a network. It takes in the input to the network, performs its operation, and returns the
 * output of the network. When a block is added as a child of another block, it becomes a
 * sub-network of that block.
 *
 * <p>The life-cycle of a block has 3 stages:
 *
 * <ul>
 *   <li>Construction
 *   <li>Initialization
 *   <li>Training
 * </ul>
 *
 * <p>Construction is the process of building the network. During this stage, blocks are created
 * with appropriate arguments and the desired network is built by adding creating a hierarchy of
 * parent and child blocks. At this stage, it is a bare-bones network. The parameter values are not
 * created and the shapes of the inputs are not known. The block is ready for initialization.
 *
 * <p>Initialization is the process of initializing all the parameters of the block and its
 * children, according to the inputs expected. It involves setting an {@link Initializer}, deciding
 * the {@link DataType}, and the shapes of the input. The parameter arrays are {@link
 * ai.djl.ndarray.NDArray} that are initialized according to the {@link Initializer} set. At this
 * stage, the block is expecting a specific type of input, and is ready to be trained.
 *
 * <p>Training is when we starting feeding the training data as input to the block, get the output,
 * and try to update parameters to learn. For more information about training, please refer the
 * javadoc at {@link ai.djl.training.Trainer}. At the end of training, a block represents a
 * fully-trained model.
 */
public interface Block {
    /**
     * Applies the operating function of the block once. This method should be called only on blocks
     * that are initialized.
     *
     * @param parameterStore the parameter store
     * @param inputs the input NDList
     * @return the output of the forward pass
     */
    default NDList forward(ParameterStore parameterStore, NDList inputs) {
        return forward(parameterStore, inputs, null);
    }

    /**
     * Applies the operating function of the block once. This method should be called only on blocks
     * that are initialized.
     *
     * @param parameterStore the parameter store
     * @param inputs the input NDList
     * @param params optional parameters
     * @return output of the forward pass
     */
    NDList forward(ParameterStore parameterStore, NDList inputs, PairList<String, Object> params);

    /**
     * Sets an {@link Initializer} to the block.
     *
     * @param initializer the initializer to set
     */
    void setInitializer(Initializer initializer);

    /**
     * Sets an {@link Initializer} to the specified direct parameter of the block, overriding the
     * initializer of the parameter, if already set.
     *
     * @param initializer the initializer to be set
     * @param paramName the name of the parameter
     */
    void setInitializer(Initializer initializer, String paramName);

    /**
     * Initializes the parameters of the block. This method must be called before calling `forward`.
     *
     * @param manager the NDManager to initialize the parameters
     * @param dataType the datatype of the parameters
     * @param inputShapes theshapes of the inputs to the block
     * @return the shapes of the outputs of the block
     */
    Shape[] initialize(NDManager manager, DataType dataType, Shape[] inputShapes);

    /**
     * Returns a boolean whether the block is initialized.
     *
     * @return whether the block is initialized
     */
    boolean isInitialized();

    /**
     * Guaranteed to throw an exception. Not yet implemented
     *
     * @param dataType the datatype to cast to
     * @throws UnsupportedOperationException always
     */
    void cast(DataType dataType);

    /**
     * Closes all the parameters of the block. All the updates made during training will be lost.
     */
    void clear();

    /**
     * Returns a {@link PairList} of input names, and shapes.
     *
     * @return the {@link PairList} of input names, and shapes
     */
    PairList<String, Shape> describeInput();

    /**
     * Returns a list of all the children of the block.
     *
     * @return the list of child blocks
     */
    BlockList getChildren();

    /**
     * Returns a list of all the direct parameters of the block.
     *
     * @return the list of {@link Parameter}
     */
    List<Parameter> getDirectParameters();

    /**
     * Returns a list of all the parameters of the block, including the parameters of its children
     * fetched recursively.
     *
     * @return the list of all parameters of the block
     */
    ParameterList getParameters();

    /**
     * Returns the shape of the specified direct parameter of this block given the shapes of the
     * input to the block.
     *
     * @param name the name of the parameter
     * @param inputShapes the shapes of the input to the block
     * @return the shape of the parameter specified
     * @throws IllegalArgumentException if the parameter name specified is invalid
     */
    Shape getParameterShape(String name, Shape[] inputShapes);

    /**
     * Returns the expected output shapes of the block for the specified input shapes.
     *
     * @param manager an NDManager
     * @param inputShapes the shapes of the inputs
     * @return the expected output shapes of the block
     */
    Shape[] getOutputShapes(NDManager manager, Shape[] inputShapes);

    /**
     * Writes the parameters of the block to the given outputStream.
     *
     * @param os the outputstream to save the parameters to
     * @throws IOException if an I/O error occurs
     */
    void saveParameters(DataOutputStream os) throws IOException;

    /**
     * Loads the parameters from the given input stream.
     *
     * @param manager an NDManager to create the parameter arrays
     * @param is the inputstream that stream the parameter values
     * @throws IOException if an I/O error occurs
     * @throws MalformedModelException if the model file is corrupted or unsupported
     */
    void loadParameters(NDManager manager, DataInputStream is)
            throws IOException, MalformedModelException;

    /**
     * Validates that actual layout matches the expected layout.
     *
     * @param expectedLayout the expected layout
     * @param actualLayout the actual Layout
     * @throws UnsupportedOperationException if the actual layout does not match the expected layout
     */
    static void validateLayout(LayoutType[] expectedLayout, LayoutType[] actualLayout) {
        if (actualLayout.length != expectedLayout.length) {
            throw new UnsupportedOperationException(
                    "Expected layout: "
                            + LayoutType.toString(expectedLayout)
                            + ", but got: "
                            + LayoutType.toString(actualLayout));
        }
        for (int i = 0; i < actualLayout.length; i++) {
            if (actualLayout[i] != LayoutType.UNKNOWN && actualLayout[i] != expectedLayout[i]) {
                throw new UnsupportedOperationException(
                        "Expected layout: "
                                + LayoutType.toString(expectedLayout)
                                + ", but got: "
                                + LayoutType.toString(actualLayout));
            }
        }
    }
}