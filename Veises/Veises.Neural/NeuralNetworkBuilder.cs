using System;
using System.Collections.Generic;
using System.Diagnostics;

namespace Veises.Neural
{
	public sealed class NeuralNetworkBuilder: INeuralNetworkBuilder
	{
		private const int MinimalLayersCount = 3;

		private readonly INeuralNetworkLayerBuilder _neuronLayerBuilder;

		public NeuralNetworkBuilder(INeuralNetworkLayerBuilder neuronLayerBuilder)
		{
			_neuronLayerBuilder = neuronLayerBuilder ?? throw new ArgumentNullException(nameof(neuronLayerBuilder));
		}

		public INeuralNetwork Build(int[] layerNeuronsCount, IActivationFunction activationFunction)
		{
			if (layerNeuronsCount == null)
				throw new ArgumentNullException(nameof(layerNeuronsCount));
			if (activationFunction == null)
				throw new ArgumentNullException(nameof(activationFunction));

			if (layerNeuronsCount.Length < MinimalLayersCount)
				throw new ArgumentException(
					$"Neural network layers count can not be less than {MinimalLayersCount}, but found {layerNeuronsCount.Length}.",
					nameof(layerNeuronsCount));

			var layers = new List<INeuralNetworkLayer>();

			INeuralNetworkLayer previousLayer = null;

			for (var layerNumber = 0; layerNumber < layerNeuronsCount.Length; layerNumber++)
			{
				var layerType = NeuronLayerType.Hidden;

				if (layerNumber == 0)
					layerType = NeuronLayerType.Input;
				else if (layerNumber == layerNeuronsCount.Length - 1)
					layerType = NeuronLayerType.Output;

				var layer = _neuronLayerBuilder.Build(
					layerType,
					layerNeuronsCount[layerNumber],
					activationFunction);

				layers.Add(layer);

				if (layerType != NeuronLayerType.Input)
				{
					NeuralNetworkAxon.Create(previousLayer, layer);
				}

				previousLayer = layer;
			}

			return new NeuralNetwork(layers, activationFunction);
		}
	}
}