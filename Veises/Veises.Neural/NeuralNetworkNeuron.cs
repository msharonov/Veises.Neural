using System;
using System.Collections.Generic;
using System.Diagnostics;
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

		public NeuralNetworkNeuron(
			IActivationFunction activationFunction,
			Bias bias)
		{
			_activationFunction = activationFunction ?? throw new ArgumentNullException(nameof(activationFunction));
			_bias = bias;

			_outputAxons = new List<INeuralNetworkAxon>();
			_inputAxons = new List<INeuralNetworkAxon>();

			Debug.WriteLine($"Neural network neuron with type {GetType().Name} was created");
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

			if (_bias != null)
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

		/// <summary>
		/// Calculate error for output layer neuron
		/// </summary>
		/// <param name="target">Desired output value for neuron</param>
		public void BackpropagateError(double target)
		{
			if (_outputAxons.Count() > 0)
				throw new ApplicationException("Can't calculate error for non-output layer neuron");

			Error = (Output - target) * _activationFunction.GetDerivative(Output);
		}

		/// <summary>
		/// Calculare error for hidder layer neuron
		/// </summary>
		public void BackpropagateError()
		{
			if (_outputAxons.Count() == 0)
				throw new ApplicationException("Can't calculate error for output layer neuron");

			var weightErrorSum = _outputAxons.Sum(_ => _.GetWeightedError());

			Error = weightErrorSum * _activationFunction.GetDerivative(Output);
		}

		/// <summary>
		/// Set value for Neuron
		/// </summary>
		/// <param name="input"></param>
		public void SetInput(double input) => Output = input;
	}
}