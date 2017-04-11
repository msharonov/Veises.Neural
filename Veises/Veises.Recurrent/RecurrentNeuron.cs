using System;
using System.Collections.Generic;
using System.Linq;
using Veises.Neural;

namespace Veises.Recurrent
{
	public sealed class RecurrentNeuron: NeuralNetworkNeuron
	{
		private readonly IReadOnlyCollection<INeuralNetworkAxon> _layerContextAxons;

		private readonly INeuralNetworkNeuron _contextNeuron;

		public RecurrentNeuron(
			IReadOnlyCollection<INeuralNetworkNeuron> layerContextNeurons,
			INeuralNetworkNeuron contextNeuron,
			IActivationFunction activationFunction,
			Bias bias)
			: base(activationFunction, bias)
		{
			if (layerContextNeurons == null)
				throw new ArgumentNullException(nameof(layerContextNeurons));

			_layerContextAxons = BuildAxonsForLayerContextNeurons(layerContextNeurons);

			_contextNeuron = contextNeuron ?? throw new ArgumentNullException(nameof(contextNeuron));
		}

		private List<INeuralNetworkAxon> BuildAxonsForLayerContextNeurons(
			IReadOnlyCollection<INeuralNetworkNeuron> layerContextNeurons)
		{
			var layerContextAxons = new List<INeuralNetworkAxon>();

			foreach (var contextNeuron in layerContextNeurons)
			{
				layerContextAxons.Add(new NeuralNetworkAxon(contextNeuron, this));
			}

			return layerContextAxons;
		}

		public override void CalculateOutput()
		{
			var inputSum = 0d;

			foreach (var inputAxon in _inputAxons)
			{
				inputSum += inputAxon.GetOutput();
			}

			inputSum += _inputAxons.Sum(_ => _.GetOutput());

			inputSum += _bias.Weight;

			inputSum += _layerContextAxons.Sum(_ => _.GetOutput());

			Output = _activationFunction.Activate(inputSum);

			_contextNeuron.SetInput(Output);
		}
	}
}