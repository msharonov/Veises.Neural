using System;
using System.Collections.Generic;
using System.Linq;
using Veises.Neural;

namespace Veises.Recurrent
{
	public sealed class RecurrentNeuralNetworkNeuron: NeuralNetworkNeuron
	{
		private readonly IReadOnlyCollection<INeuralNetworkAxon> _layerContextAxons;

		private readonly INeuralNetworkAxon _outputContextAxon;

		public RecurrentNeuralNetworkNeuron(
			IReadOnlyCollection<INeuralNetworkNeuron> layerContextNeurons,
			INeuralNetworkNeuron outputContextNeuron,
			IActivationFunction activationFunction,
			Bias bias)
			: base(activationFunction, bias)
		{
			if (layerContextNeurons == null)
				throw new ArgumentNullException(nameof(layerContextNeurons));
			if (outputContextNeuron == null)
				throw new ArgumentNullException(nameof(outputContextNeuron));

			_layerContextAxons = BuildAxonsForLayerContextNeurons(layerContextNeurons);

			_outputContextAxon = new NeuralnetworkStaticAxon(this, outputContextNeuron);

			Output = 0.5d;
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
			var inputAxons = _inputAxons
				.Where(axon => !_layerContextAxons.Contains(axon));

			var inputSum = inputAxons.Sum(_ => _.GetOutput());

			var contextInputSum = _layerContextAxons.Sum(_ => _.GetOutput());

			inputSum += contextInputSum;

			inputSum += _bias.Weight;

			Output = _activationFunction.Activate(inputSum);
		}
	}
}