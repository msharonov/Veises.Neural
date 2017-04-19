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
			NeuralNetworkBias bias)
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
				contextNeuron.BackpropagateError();
			}
		}

		public override void CalculateOutputs()
		{
			base.CalculateOutputs();

			CloneLayerOutputToContext();
		}

		private void CloneLayerOutputToContext()
		{
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

			return Softmax(outputs).ToList();
		}

		private static IEnumerable<double> Softmax(IEnumerable<double> inputValues)
		{
			if (inputValues == null)
				throw new ArgumentNullException(nameof(inputValues));

			var sum = inputValues.Sum(_ => Math.Exp(_));

			foreach (var inputValue in inputValues)
			{
				yield return Math.Exp(inputValue) / sum;
			}
		}
	}
}