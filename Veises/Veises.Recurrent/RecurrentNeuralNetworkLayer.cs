using System;
using System.Collections.Generic;
using System.Linq;
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

		public override void AdjustWeights()
		{
			base.AdjustWeights();

			foreach (var contextNeuron in _contextNeurons)
			{
				contextNeuron.AdjustWeights();
			}
		}

		public override void BackpropagateError()
		{
			base.BackpropagateError();

			foreach (var contextNeuron in _contextNeurons)
			{
				contextNeuron.CalculateError();
			}
		}

		public override void CalculateOutputs()
		{
			base.CalculateOutputs();

			var contextNeuronsEnumerator = _contextNeurons.GetEnumerator();

			contextNeuronsEnumerator.MoveNext();

			foreach (var neuron in Neurons)
			{
				contextNeuronsEnumerator.Current.SetInput(neuron.Output);

				contextNeuronsEnumerator.MoveNext();
			}
		}

		public override IReadOnlyCollection<double> GetOutputs()
		{
			var outputs = base.GetOutputs();

			if (LayerType != NeuronLayerType.Output)
				return outputs;

			return Softmax(outputs.ToArray());
		}

		private double[] Softmax(double[] input)
		{
			var result = new double[input.Length];

			var sum = 0.0;

			for (var k = 0; k < input.Length; ++k)
				sum += Math.Exp(input[k]);

			for (var k = 0; k < input.Length; ++k)
				result[k] = Math.Exp(input[k]) / sum;

			return result;
		}
	}
}