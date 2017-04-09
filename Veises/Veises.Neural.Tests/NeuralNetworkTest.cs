using System;
using NUnit.Framework;
using FluentAssertions;
using System.Linq;

namespace Veises.Neural.Tests
{
	[TestFixture]
	public class NeuralNetworkTest
	{
		private INeuralNetwork _target;

		private static readonly double[] OneInput =
		{
			0, 1, 0,
			0, 1, 0,
			0, 1, 0,
			0, 1, 0,
			1, 1, 1
		};

		private static readonly double[] OneOutput =
		{
			1, 0, 0
		};

		private static readonly double[] FourInput =
		{
			1, 0, 1,
			1, 0, 1,
			1, 1, 1,
			0, 0, 1,
			0, 0, 1
		};

		private static readonly double[] FourOutput =
		{
			0, 1, 0
		};

		private static readonly double[] FiveInput =
		{
			1, 1, 1,
			1, 0, 0,
			1, 1, 1,
			0, 0, 1,
			1, 1, 1,
		};

		private static readonly double[] FiveOutput =
		{
			0, 0, 1
		};

		public static object[] TestCases =
		{
			new object[] {OneInput, OneOutput},
			new object[] {FourInput, FourOutput},
			new object[] {FiveInput, FiveOutput}
		};

		[SetUp]
		public void SetUp()
		{
			var neuronBuilder = new NeuronBuilder(new SigmoidFunction());
			var neuralNetworkLayerBuilder = new NeuralNetworkLayerBuilder(neuronBuilder);
			var neuralNetworkBuilder = new NeuralNetworkBuilder(neuralNetworkLayerBuilder);

			_target = neuralNetworkBuilder.Build(new[] { 15, 50, 50, 3 });

			var networkTrainer = new NeuralNetworkTrainer();

			networkTrainer.Load(_target);
			networkTrainer.Train(
				new NetworkLearnCase(OneInput, OneOutput),
				new NetworkLearnCase(FourInput, FourOutput),
				new NetworkLearnCase(FiveInput, FiveOutput));
		}

		[TestCaseSource(nameof(TestCases))]
		public void ShouldRecognizeSymbolsViaNet(double[] input, double[] output)
		{
			var result = _target.GetOutputs(input).ToArray();

			for (var i = 0; i < output.Length; i++)
			{
				output[i].Should().BeApproximately(result[i], 0.05d);
			}
		}
	}
}