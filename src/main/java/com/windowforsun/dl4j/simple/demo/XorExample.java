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

public class XorExample {

	public static void main(String[] args) {
		// 1. 신경망 구성 정의
		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
			.updater(new Sgd(0.1))
			.list()
			.layer(new DenseLayer.Builder()
				.nIn(2) // 입력 노드 개수
				.nOut(3) // 은닉층 노드 개수
				.activation(Activation.RELU) // 활성화 함수
				.build())
			.layer(new OutputLayer.Builder(LossFunctions.LossFunction.MSE) // 출력층
				.nIn(3) // 은닉층 노드 개수
				.nOut(1) // 출력 노드 개수
				.activation(Activation.IDENTITY) // 활성화 함수
				.build())
			.build();

		// 2. 모델 초기화
		MultiLayerNetwork model = new MultiLayerNetwork(conf);
		model.init();

		// 3. 학습 데이터 생성 (XOR 문제)
		double[][] input = {
			{0, 0},
			{0, 1},
			{1, 0},
			{1, 1}
		};
		double[][] labels = {
			{0},
			{1},
			{1},
			{0}
		};

		DataSet dataset = new DataSet(Nd4j.create(input), Nd4j.create(labels));

		// 4. 모델 학습
		for (int i = 0; i < 1000; i++) { // 1000번의 에포크 학습
			model.fit(dataset);
		}

		// 5. 예측 결과 출력
		System.out.println("예측 결과:");
		for (double[] sample : input) {
			double[] prediction = model.output(Nd4j.create(new double[][] {sample})).toDoubleVector();
			System.out.println(Nd4j.create(sample) + " -> " + Nd4j.create(prediction));
		}
	}
}
