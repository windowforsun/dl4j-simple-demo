package com.windowforsun.dl4j.simple.demo;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class LinearRegressionExample {

	public static void main(String[] args) {
		// 1. 신경망 구성 정의
		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
			.updater(new Sgd(0.001)) // 학습률
			.list()
			.layer(new DenseLayer.Builder()
				.nIn(1) // 입력 노드 개수 (x)
				.nOut(10) // 은닉층 노드 개수
				.activation(Activation.RELU) // 활성화 함수
				.build())
			.layer(new OutputLayer.Builder(LossFunctions.LossFunction.MSE) // 출력층
				.nIn(10) // 은닉층 노드 개수
				.nOut(1) // 출력 노드 개수 (y)
				.activation(Activation.IDENTITY) // 선형 활성화 함수
				.build())
			.build();

		// 2. 모델 초기화
		MultiLayerNetwork model = new MultiLayerNetwork(conf);
		model.init();

		// 3. 학습 데이터 생성
		int numSamples = 100; // 데이터 샘플 수
		double[][] input = new double[numSamples][1]; // 입력 데이터
		double[][] labels = new double[numSamples][1]; // 출력 데이터
		for (int i = 0; i < numSamples; i++) {
			double x = i * 0.1; // 입력값 생성
			// double x = i; // 입력값 생성
			input[i][0] = x;
			labels[i][0] = 2 * x + 1; // y = 2 * x + 1
		}
		DataSet dataset = new DataSet(Nd4j.create(input), Nd4j.create(labels));

		// 4. 모델 학습
		int numEpochs = 1000; // 학습 에포크 수
		for (int i = 0; i < numEpochs; i++) {
			model.fit(dataset);
		}

		// 5. 예측 결과 출력
		System.out.println("예측 결과:");
		double[][] testInputs = {{0d}, {1d}, {2d}, {3d}, {4d}}; // 테스트 입력값
		for (double[] testInput : testInputs) {
			double[] prediction = model.output(Nd4j.create(new double[][] {testInput})).toDoubleVector();
			System.out.printf("입력: %.1f -> 예측 출력: %.3f\n", testInput[0], prediction[0]);
		}
	}
}
