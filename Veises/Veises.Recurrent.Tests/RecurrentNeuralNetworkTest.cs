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
				3, 50, 50, 1);
		}

		private static NetworkLearnCase GetLearnCase(double step)
		{
			var testInput = new[]
			{
				0.1d + step * 0.1d,
				0.2d + step * 0.1d,
				0.3d + step * 0.1d
			};

			var desiredOutput = new[]
			{
				Math.Sin(0.4d + step * 0.1d)
			};

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
				GetLearnCase(2d));

			var output = _recurrentNetwork
				.GetOutputs()
				.ToArray();

			var validationInput = GetLearnCase(3d);

			_recurrentNetwork.SetInputs(validationInput.Input);

			// --

			var result = _recurrentNetwork.GetOutputs();

			// --
		}
	}
}