using NUnit.Framework;
using System;
using System.Linq;
using Veises.Neural;

namespace Veises.Recurrent.Tests
{
	[TestFixture]
	public sealed class RecurrentNeuralNetworkTest
	{
		private INeuralNetwork _recurrentNetwork;

		[SetUp]
		public void SetUp()
		{
			var neuronNetworkBuilder = new RecurrentNeuralNetworkBuilder(
				new NeuralNetworkLayerBuilder(),
				new RecurrentNetworkLayerBuilder());

			_recurrentNetwork = neuronNetworkBuilder.Build(
				new SigmoidFunction(),
				3, 3, 3);
		}

		private static NetworkLearnCase GetLearnCase(double step)
		{
			var testInput = new[]
{
				0.1d + step * 0.1d,
				0.2d + step * 0.1d,
				0.3d + step * 0.1d
			};

			var desiredOutput = testInput
				.Select(_ => Math.Sin(_))
				.ToArray();

			return new NetworkLearnCase(testInput, desiredOutput);
		}

		[Test]
		public void ShouldBuildRecurrentNetwork()
		{
			var networkTrainer = new GlobalErrorNeuralNetworkTrainer();
			networkTrainer.Load(_recurrentNetwork);

			networkTrainer.Train(
				GetLearnCase(0d),
				GetLearnCase(1d),
				GetLearnCase(2d),
				GetLearnCase(3d));

			var output = _recurrentNetwork
				.GetOutputs()
				.ToArray();

			var validationInput = new[]
			{
				0.7d,
				0.8d,
				0.9d
			};

			_recurrentNetwork.SetInputs(validationInput);

			// --

			var result = _recurrentNetwork.GetOutputs();

			// --
		}
	}
}