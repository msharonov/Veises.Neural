using System;
using System.Collections.Generic;
using System.Diagnostics;

namespace Veises.Neural
{
	public sealed class NeuralNetworkBuilder: INeuralNetworkBuilder<NeuralNetwork>
	{
		private const int MinimalLayersCount = 3;

		public NeuralNetwork Build(int[] layerNeuronsCount, IErrorFunction errorFunction)
		{
			if (layerNeuronsCount == null)
				throw new ArgumentNullException(nameof(layerNeuronsCount));

			if (errorFunction == null)
				throw new ArgumentNullException(nameof(errorFunction));

			if (layerNeuronsCount.Length < MinimalLayersCount)
				throw new ArgumentException(
					$"Neuron layers count can not be less than {MinimalLayersCount}, but found {layerNeuronsCount.Length}.",
					nameof(layerNeuronsCount));

			var layers = new List<NeuronLayer>();

			for (var layerNumber = 0; layerNumber < layerNeuronsCount.Length; layerNumber++)
			{
				var layerType = NeuronLayerType.Hidden;

				if (layerNumber == 0)
					layerType = NeuronLayerType.Input;
				else if (layerNumber == layerNeuronsCount.Length - 1)
					layerType = NeuronLayerType.Output;

				var layer = NeuronLayer.Create(layerType, layerNeuronsCount[layerNumber]);

				layers.Add(layer);

				if (layerNumber > 0)
				{
					var previousLayer = layers[layerNumber - 1];

					Axon.Create(previousLayer, layer);
				}

				Debug.WriteLine($"Neuron {layerType} layer {layerNumber + 1} created");
			}

			return new NeuralNetwork(layers, errorFunction);
		}
	}
}