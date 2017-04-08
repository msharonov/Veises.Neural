using System;
using System.Collections.Generic;

namespace Veises.Neural
{
	public class Neuron
	{
		protected readonly IActivationFunction _activationFunction;

		protected readonly IList<Axon> _outputAxons;

		protected readonly IList<Axon> _inputAxons;

		protected readonly Bias _bias;

		public double Error { get; protected set; }

		public double Output { get; protected set; }

		public Neuron(IActivationFunction activationFunction, Bias bias)
		{
			_activationFunction = activationFunction ?? throw new ArgumentNullException(nameof(activationFunction));
			_bias = bias ?? throw new ArgumentNullException(nameof(bias));

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

		public virtual void CalculateOutput()
		{
			var inputSum = 0d;

			foreach (var axon in _inputAxons)
			{
				inputSum += axon.GetOutput();
			}

			inputSum += _bias.Weight;

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