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

			var neuralNetworkLayerBuilder = new RecurrentNetworkLayerBuilder(activationFunction);
			var neuronNetworkBuilder = new NeuralNetworkBuilder(neuralNetworkLayerBuilder, activationFunction);

			_recurrentNetwork = neuronNetworkBuilder.Build(new[] { 3, 3, 3 });
		}
		
		[Test]
		public void ShouldBuildRecurrentNetwork()
		{
			var networkTrainer = new NeuralNetworkTrainer();
			networkTrainer.Load(_recurrentNetwork);

			var testInput = new[]
			{
				0.1d,
				0.2d,
				0.3d
			};

			var desiredOutput = testInput
				.Select(_ => Math.Sin(_))
				.ToArray();

			networkTrainer.Train(new NetworkLearnCase(testInput, desiredOutput));

//			testInput = new[]
//{
//				2d,
//				3d,
//				4d
//			};

//			desiredOutput = testInput
//				.Select(_ => Math.Sin(_))
//				.ToArray();

//			networkTrainer.Train(new NetworkLearnCase(testInput, desiredOutput));

//			testInput = new[]
//{
//				3d,
//				4d,
//				5d
//			};

//			desiredOutput = testInput
//				.Select(_ => Math.Sin(_))
//				.ToArray();

//			networkTrainer.Train(new NetworkLearnCase(testInput, desiredOutput));

			var validationInput = new[]
			{
				0.2d,
				0.3d,
				0.4d
			};

			_recurrentNetwork.SetInputs(validationInput);

			// --

			var result = _recurrentNetwork.GetOutputs();

			// --
		}
	}
}