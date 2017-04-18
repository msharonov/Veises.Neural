using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using Veises.Neural;

namespace Veises.Recurrent
{
	public sealed class RecurrentNetworkLayerBuilder: INeuralNetworkLayerBuilder
	{
		private const int MinimalNeuronsCount = 1;

		public INeuralNetworkLayer Build(
			NeuronLayerType layerType,
			int neuronsCount,
			IActivationFunction activationFunction)
		{
			if (neuronsCount < MinimalNeuronsCount)
				throw new ArgumentException($"Layer neurons count can not be less than {MinimalNeuronsCount}");

			var contextNeurons = Enumerable.Range(0, neuronsCount)
				.Select(_ => new NeuralNetworkNeuron(activationFunction, null))
				.ToList();

			var layerBias = new Bias();

			var layerNeurons = Enumerable.Range(0, neuronsCount)
				.Select(_ => new RecurrentNeuralNetworkNeuron(
					contextNeurons,
					contextNeurons[_],
					activationFunction,
					layerBias))
				.ToList();

			var recurrentLayer = new RecurrentNeuralNetworkLayer(
				layerType,
				layerNeurons,
				contextNeurons,
				layerBias);

			Debug.WriteLine($"Recurrent {layerType} layer with {neuronsCount} neurons was created");

			return recurrentLayer;
		}
	}
}