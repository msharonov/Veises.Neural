using System;
using System.Collections.Generic;
using Veises.Neural;

namespace Veises.Recurrent
{
	public sealed class RecurrentNeuralNetworkLayer: NeuralNetworkLayer
	{
		private IReadOnlyCollection<INeuralNetworkNeuron> _contextNeurons;

		public RecurrentNeuralNetworkLayer(
			NeuronLayerType layerType,
			IReadOnlyCollection<INeuralNetworkNeuron> layerNeurons,
			IReadOnlyCollection<INeuralNetworkNeuron> contextNeurons,
			Bias bias)
			: base(layerType, layerNeurons, bias)
		{
			_contextNeurons = contextNeurons ?? throw new ArgumentNullException(nameof(contextNeurons));
		}
	}
}