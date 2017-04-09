using System;
using System.Collections.Generic;

namespace Veises.Neural
{
	public class NeuralNetworkNeuron: INeuralNetworkNeuron
	{
		protected readonly IActivationFunction _activationFunction;

		protected readonly IList<INeuralNetworkAxon> _outputAxons;

		protected readonly IList<INeuralNetworkAxon> _inputAxons;

		protected readonly Bias _bias;

		public double Error { get; protected set; }

		public double Output { get; protected set; }

		public NeuralNetworkNeuron(IActivationFunction activationFunction, Bias bias)
		{
			_activationFunction = activationFunction ?? throw new ArgumentNullException(nameof(activationFunction));
			_bias = bias ?? throw new ArgumentNullException(nameof(bias));

			_outputAxons = new List<INeuralNetworkAxon>();
			_inputAxons = new List<INeuralNetworkAxon>();
		}

		public void AddOutput(INeuralNetworkAxon axon)
		{
			if (axon == null)
				throw new ArgumentNullException(nameof(axon));

			_outputAxons.Add(axon);
		}

		public void AddInput(INeuralNetworkAxon axon)
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

		public void CalculateError(double errorTerm) =>
			Error = errorTerm * _activationFunction.Deactivate(Output);

		public void SetInput(double input)
		{
			if (_inputAxons.Count > 0)
				throw new ArgumentException("Input value can not be set for a non-input layer neurons.");

			Output = input;
		}
	}
}