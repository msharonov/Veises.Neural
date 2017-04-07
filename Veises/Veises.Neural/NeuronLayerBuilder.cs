using System;
using System.Linq;

namespace Veises.Neural
{
	public sealed class NeuronLayerBuilder: INeuronLayerBuilder
	{
		private readonly INeuronBuilder _neuronBuilder;

		public NeuronLayerBuilder(INeuronBuilder neuronBuilder)
		{
			_neuronBuilder = neuronBuilder ?? throw new ArgumentNullException(nameof(neuronBuilder));
		}

		public NeuronLayer Build(NeuronLayerType layerType, int neuronsCount)
		{
			if (neuronsCount < 1)
				throw new ArgumentException("Layer neurons count can not be less than 1.");

			var layerBias = new Bias();

			var neurons = Enumerable
				.Range(0, neuronsCount)
				.Select(_ => _neuronBuilder.Build(layerBias));

			return new NeuronLayer(layerType, neurons, layerBias);
		}
	}
}