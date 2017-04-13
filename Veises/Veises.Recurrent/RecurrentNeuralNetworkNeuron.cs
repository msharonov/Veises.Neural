using System;
using System.Collections.Generic;
using System.Linq;
using Veises.Neural;

namespace Veises.Recurrent
{
	public sealed class RecurrentNeuralNetworkNeuron: NeuralNetworkNeuron
	{
		private readonly IReadOnlyCollection<INeuralNetworkAxon> _layerContextAxons;

		public RecurrentNeuralNetworkNeuron(
			IReadOnlyCollection<INeuralNetworkNeuron> layerContextNeurons,
			IActivationFunction activationFunction,
			Bias bias)
			: base(activationFunction, bias)
		{
			if (layerContextNeurons == null)
				throw new ArgumentNullException(nameof(layerContextNeurons));

			_layerContextAxons = BuildAxonsForLayerContextNeurons(layerContextNeurons);
		}

		private IReadOnlyCollection<INeuralNetworkAxon> BuildAxonsForLayerContextNeurons(
			IEnumerable<INeuralNetworkNeuron> layerContextNeurons)
		{
			var layerContextAxons = new List<INeuralNetworkAxon>();

			foreach (var contextNeuron in layerContextNeurons)
			{
				layerContextAxons.Add(NeuralNetworkAxon.Create(contextNeuron, this));
			}

			return layerContextAxons;
		}

		public override void CalculateOutput()
		{
			var inputSum = _inputAxons.Sum(_ => _.GetOutput());

			inputSum += _bias.Weight;

			var contextInputSum = _layerContextAxons.Sum(_ => _.GetOutput());

			inputSum += contextInputSum;

			Output = _activationFunction.Activate(inputSum);
		}
	}
}