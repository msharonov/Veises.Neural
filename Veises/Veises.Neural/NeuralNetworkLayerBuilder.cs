using System;
using System.Linq;

namespace Veises.Neural
{
	public sealed class NeuralNetworkLayerBuilder: INeuralNetworkLayerBuilder
	{
		private const int LayerMinimalNeuronsCount = 1;

		private readonly IActivationFunction _activationFunction;

		public NeuralNetworkLayerBuilder(IActivationFunction activationFunction)
		{
			_activationFunction = activationFunction ?? throw new ArgumentNullException(nameof(activationFunction));
		}

		public INeuralNetworkLayer Build(NeuronLayerType layerType, int neuronsCount)
		{
			if (neuronsCount < LayerMinimalNeuronsCount)
				throw new ArgumentException($"Layer neurons count can not be less than {LayerMinimalNeuronsCount}.");

			var layerBias = new Bias();

			var neurons = Enumerable
				.Range(0, neuronsCount)
				.Select(_ => new NeuralNetworkNeuron(_activationFunction, layerBias));

			return new NeuralNetworkLayer(layerType, neurons, layerBias);
		}
	}
}