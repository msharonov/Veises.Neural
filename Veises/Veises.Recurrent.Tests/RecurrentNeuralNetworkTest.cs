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
			var neuronBuilder = new NeuronBuilder(new TanhActivationFunction());
			var neuralNetworkLayerBuilder = new NeuralNetworkLayerBuilder(neuronBuilder);
			var recurrentNeuronNetworkBuilder = new NeuralNetworkBuilder(neuralNetworkLayerBuilder);

			_recurrentNetwork = recurrentNeuronNetworkBuilder.Build(new[] { 1, 1, 1 });
		}
		
		[TestMethod]
		public void ShouldBuildRecurrentNetwork()
		{
		}
	}
}