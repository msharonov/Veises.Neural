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
			var activationFunction = new SigmoidFunction();

			var neuralNetworkLayerBuilder = new NeuralNetworkLayerBuilder(activationFunction);
			var recurrentNeuronNetworkBuilder = new NeuralNetworkBuilder(neuralNetworkLayerBuilder, activationFunction);

			_recurrentNetwork = recurrentNeuronNetworkBuilder.Build(new[] { 3, 3, 3 });
		}
		
		[Test]
		public void ShouldBuildRecurrentNetwork()
		{
			var networkTrainer = new NeuralNetworkTrainer();

			var testInput = new[]
			{
				1d,
				2d,
				3d
			};

			var desiredOutput = testInput
				.Select(_ => Math.Sin(_))
				.ToArray();

			networkTrainer.Load(_recurrentNetwork);
			networkTrainer.Train(
				new NetworkLearnCase(testInput, desiredOutput));

			var validationInput = new[]
			{
				2d,
				3d,
				4d
			};

			_recurrentNetwork.SetInputs(validationInput);

			// --

			var result = _recurrentNetwork.GetOutputs();

			// --
		}
	}
}