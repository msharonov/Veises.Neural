using System;
using System.Collections.Generic;
using System.Linq;

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

		public void CalculateError(double target)
		{
			if (_outputAxons.Count() > 0)
				throw new ApplicationException("Can't calculate error for non-output layer neuron");

			var dif = target - Output;

			var errorSum = 0d;

			foreach (var outputAxon in _outputAxons)
			{
				errorSum += outputAxon.WeightedError;
			}

			Error = dif * _activationFunction.Deactivate(errorSum);
		}

		public void CalculateError()
		{
			if (_outputAxons.Count() == 0)
				throw new ApplicationException("Can't calculate error for output layer neuron");

			var weightErrorSum = 0d;

			foreach (var axon in _outputAxons)
			{
				weightErrorSum += axon.WeightedError;
			}

			Error = weightErrorSum * _activationFunction.Deactivate(Output);
		}

		public void SetInput(double input) => Output = input;
	}
}