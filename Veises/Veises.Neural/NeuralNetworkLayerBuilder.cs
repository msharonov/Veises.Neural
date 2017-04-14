using System;
using System.Diagnostics;
using System.Linq;

namespace Veises.Neural
{
	public sealed class NeuralNetworkLayerBuilder: INeuralNetworkLayerBuilder
	{
		private const int LayerMinimalNeuronsCount = 1;

		public INeuralNetworkLayer Build(
			NeuronLayerType layerType,
			int neuronsCount,
			IActivationFunction activationFunction)
		{
			if (activationFunction == null)
				throw new ArgumentNullException(nameof(activationFunction));

			if (neuronsCount < LayerMinimalNeuronsCount)
				throw new ArgumentException($"Layer neurons count can not be less than {LayerMinimalNeuronsCount}.");

			var layerBias = new Bias();

			var neurons = Enumerable
				.Range(0, neuronsCount)
				.Select(_ => new NeuralNetworkNeuron(activationFunction, layerBias));

			var layer = new NeuralNetworkLayer(layerType, neurons, layerBias);

			Debug.WriteLine($"Network {layerType} layer with {neuronsCount} neurons was created");

			return layer;
		}
	}
}