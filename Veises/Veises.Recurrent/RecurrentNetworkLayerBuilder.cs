using System;
using System.Collections.Generic;
using System.Linq;
using Veises.Neural;

namespace Veises.Recurrent
{
	public sealed class RecurrentNetworkLayerBuilder: INeuralNetworkLayerBuilder
	{
		private const int MinimalNeuronsCount = 1;

		private readonly IActivationFunction _activationFunction;

		public RecurrentNetworkLayerBuilder(IActivationFunction activationFunction)
		{
			_activationFunction = activationFunction ?? throw new ArgumentNullException(nameof(activationFunction));
		}

		public INeuralNetworkLayer Build(NeuronLayerType layerType, int neuronsCount)
		{
			if (neuronsCount < MinimalNeuronsCount)
				throw new ArgumentException($"Layer neurons count can not be less than {MinimalNeuronsCount}");

			var contextNeurons = Enumerable.Range(0, neuronsCount)
				.Select(_ => new NeuralNetworkNeuron(_activationFunction, null))
				.ToList();

			var layerBias = new Bias();

			var layerNeurons = new List<RecurrentNeuron>();

			for (var i = 0; i < neuronsCount; i++)
			{
				var layerNeuron = new RecurrentNeuron(
					contextNeurons,
					contextNeurons[i],
					_activationFunction,
					layerBias);
			}

			var layer = new RecurrentNeuralNetworkLayer(
				layerType,
				layerNeurons,
				contextNeurons,
				layerBias);

			return layer;
		}
	}
}