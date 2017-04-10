using Microsoft.VisualStudio.TestTools.UnitTesting;
using Veises.Neural;

namespace Veises.Recurrent.Tests
{
	[TestClass]
	public sealed class RecurrentNeuralNetworkTest
	{
		private INeuralNetwork _recurrentNetwork;
		
		[TestInitialize]
		public void SetUp()
		{
			var activationFunction = new SigmoidFunction();

			var neuronBuilder = new NeuronBuilder(activationFunction);
			var neuralNetworkLayerBuilder = new NeuralNetworkLayerBuilder(neuronBuilder);
			var recurrentNeuronNetworkBuilder = new NeuralNetworkBuilder(neuralNetworkLayerBuilder, activationFunction);

			_recurrentNetwork = recurrentNeuronNetworkBuilder.Build(new[] { 1, 1, 1 });
		}
		
		[TestMethod]
		public void ShouldBuildRecurrentNetwork()
		{
		}
	}
}