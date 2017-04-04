using System;
using System.Collections.Generic;

namespace Veises.Neural
{
	public sealed class Neuron
	{
		private readonly IActivationFunction _activationFunction;

		private readonly IList<Axon> _outputAxons;

		private readonly IList<Axon> _inputAxons;

		private readonly Bias _layerBias;

		public double Error { get; private set; }

		public double Output { get; private set; }

		public Neuron(IActivationFunction activationFunction, Bias bias)
		{
			_activationFunction = activationFunction ?? throw new ArgumentNullException(nameof(activationFunction));
			_layerBias = bias ?? throw new ArgumentNullException(nameof(bias));

			_outputAxons = new List<Axon>();
			_inputAxons = new List<Axon>();
		}

		public void AddOutput(Axon axon)
		{
			if (axon == null)
				throw new ArgumentNullException(nameof(axon));

			_outputAxons.Add(axon);
		}

		public void AddInput(Axon axon)
		{
			if (axon == null)
				throw new ArgumentNullException(nameof(axon));

			_inputAxons.Add(axon);
		}

		public void CalculateOutput()
		{
			var inputSum = 0d;

			foreach (var axon in _inputAxons)
			{
				inputSum += axon.GetOutput();
			}

			inputSum += _layerBias.Weight;

			Output = _activationFunction.Activate(inputSum);
		}

		public void AdjustWeights()
		{
			foreach (var axon in _outputAxons)
			{
				axon.AdjustWeight();
			}
		}

		public void CalculateError()
		{
			var weightErrorSum = 0d;

			foreach (var axon in _outputAxons)
			{
				weightErrorSum += axon.WeightedError;
			}

			CalculateError(weightErrorSum);
		}

		public void CalculateError(double errorTerm) => Error = errorTerm * (Output * (1 - Output));

		public void SetInput(double input)
		{
			if (_inputAxons.Count > 0)
				throw new ArgumentException("Input value can not be set for a non-input layer neurons.");

			Output = input;
		}
	}
}