using System;
using System.Linq;

namespace Veises.Neural
{
	public sealed class NeuralNetworkLayerBuilder: INeuralNetworkLayerBuilder
	{
		private readonly INeuronBuilder _neuronBuilder;

		private const int LayerMinimalNeuronsCount = 1;

		public NeuralNetworkLayerBuilder(INeuronBuilder neuronBuilder)
		{
			_neuronBuilder = neuronBuilder ?? throw new ArgumentNullException(nameof(neuronBuilder));
		}

		public INeuralNetworkLayer Build(NeuronLayerType layerType, int neuronsCount)
		{
			if (neuronsCount < LayerMinimalNeuronsCount)
				throw new ArgumentException($"Layer neurons count can not be less than {LayerMinimalNeuronsCount}.");

			var layerBias = new Bias();

			var neurons = Enumerable
				.Range(0, neuronsCount)
				.Select(_ => _neuronBuilder.Build(layerBias));

			return new NeuronLayer(layerType, neurons, layerBias);
		}
	}
}