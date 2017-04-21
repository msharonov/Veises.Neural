using System;
using System.Collections.Generic;
using Veises.Neural;

namespace Veises.Recurrent
{
	public sealed class RecurrentNeuralNetworkBuilder: INeuralNetworkBuilder
	{
		private const int MinimalLayersCount = 3;

		private readonly INeuralNetworkLayerBuilder _layerBuilder;

		private readonly INeuralNetworkLayerBuilder _recurrentLayerBuilder;

		public RecurrentNeuralNetworkBuilder(
			INeuralNetworkLayerBuilder layerBuilder,
			INeuralNetworkLayerBuilder recurrentLayerBuilder)
		{
			_layerBuilder = layerBuilder ?? throw new ArgumentNullException(nameof(layerBuilder));
			_recurrentLayerBuilder = recurrentLayerBuilder ?? throw new ArgumentNullException(nameof(layerBuilder));
		}

		public INeuralNetwork Build(
			IActivationFunction activationFunction,
			params int[] layerNeuronsCount)
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

			var inputLayer = _layerBuilder.Build(
				NeuronLayerType.Input,
				layerNeuronsCount[0],
				activationFunction);

			layers.Add(inputLayer);

			var previousLayer = inputLayer;

			for (var i = 0; i < layerNeuronsCount.Length - 2; i ++)
			{
				var recurrentLayer = _recurrentLayerBuilder.Build(
					NeuronLayerType.Hidden,
					layerNeuronsCount[i + 1],
					activationFunction);

				layers.Add(recurrentLayer);

				NeuralNetworkAxon.Create(previousLayer, recurrentLayer);

				previousLayer = recurrentLayer;
			}

			var outputLayer = _layerBuilder.Build(
				NeuronLayerType.Output,
				layerNeuronsCount[layerNeuronsCount.Length - 1],
				activationFunction);

			layers.Add(outputLayer);

			NeuralNetworkAxon.Create(previousLayer, outputLayer);

			return new NeuralNetwork(layers, activationFunction);
		}
	}
}