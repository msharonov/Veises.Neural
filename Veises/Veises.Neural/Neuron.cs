using System;
using System.Collections.Generic;

namespace Veises.Neural
{
	public sealed class Neuron
	{
		private readonly IActivationFunction _activationFunction;

		private readonly IList<Axon> _outputAxons;

		private readonly IList<Axon> _inputAxons;

		public readonly Bias Bias;

		public double Error { get; private set; }

		public double Output { get; private set; }

		public Neuron(IActivationFunction activationFunction, Bias bias)
		{
			_activationFunction = activationFunction ?? throw new ArgumentNullException(nameof(activationFunction));
			Bias = bias ?? throw new ArgumentNullException(nameof(bias));

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

		public double CalculateOutput()
		{
			var inputSum = 0d;

			foreach (var axon in _inputAxons)
			{
				inputSum += axon.GetOutput();
			}

			inputSum += Bias.Value;

			return Output = _activationFunction.Activate(inputSum, Bias.Value);
		}

		public void SetOutput(double output)
		{
			if (_inputAxons.Count > 0d)
				throw new ArgumentException(nameof(output));

			Output = output;
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
				throw new ArgumentException(nameof(input));

			Output = input;
		}
	}
}