package com.windowforsun.dl4j.simple.demo.test;

import java.util.List;

import org.deeplearning4j.datasets.iterator.utilty.ListDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class LinearRegressionExam {

    public static void main(String[] args) {

        // 데이터 생성
        int numSamples = 100;
        double[] x = new double[numSamples];
        double[] y = new double[numSamples];
        for (int i = 0; i < numSamples; i++) {
            x[i] = i;
            y[i] = 2 * x[i] + 1;
        }

        // 데이터 정규화
        double mean = Nd4j.mean(Nd4j.create(x)).getDouble(0);
        double std = Nd4j.std(Nd4j.create(x)).getDouble(0);
        for (int i = 0; i < numSamples; i++) {
            x[i] = (x[i] - mean) / std;
            y[i] = (y[i] - mean) / std;
        }

        // 데이터셋 생성
        INDArray input = Nd4j.create(x, new int[]{numSamples, 1});
        INDArray labels = Nd4j.create(y, new int[]{numSamples, 1});
        DataSet dataSet = new DataSet(input, labels);
        List<DataSet> listDs = dataSet.asList();
        DataSetIterator iterator = new ListDataSetIterator<>(listDs, numSamples);

        // 신경망 구성 설정
        MultiLayerConfiguration conf = new org.deeplearning4j.nn.conf.NeuralNetConfiguration.Builder()
                .updater(new Sgd(0.001))
                .list()
                .layer(0, new DenseLayer.Builder().nIn(1).nOut(1)
                        .activation(Activation.IDENTITY)
                        .build())
                .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .activation(Activation.IDENTITY)
                        .nIn(1).nOut(1).build())
                .build();

        // 모델 생성
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(10));

        // 모델 학습
        int nEpochs = 2000;
        for (int i = 0; i < nEpochs; i++) {
            iterator.reset();
            model.fit(iterator);
        }

        // 예측
        double[] testXArray = new double[]{0d, 1d, 2d, 3d, 4d, 5d, 6d, 7d, 8d, 9d, 10d};
        // double testX = 4.0;
        for (double testX : testXArray) {
            double testXNorm = (testX - mean) / std;
            INDArray inputTest = Nd4j.create(new double[]{testXNorm}, new int[]{1, 1});
            INDArray output = model.output(inputTest, false);
            double predictedY = output.getDouble(0) * std + mean;
            System.out.println("Predicted value for input " + testX + " is " + predictedY);
        }
        // double testXNorm = (testX - mean) / std;
        // INDArray inputTest = Nd4j.create(new double[]{testXNorm}, new int[]{1, 1});
        // INDArray output = model.output(inputTest, false);
        // double predictedY = output.getDouble(0) * std + mean;
        // System.out.println("Predicted value for input " + testX + " is " + predictedY);
    }
}