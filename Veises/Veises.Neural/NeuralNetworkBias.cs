using System;
using System.Collections.Generic;

namespace Veises.Neural
{
	public sealed class NeuralNetworkBias: INeuralNetworkNeuron
	{
		private readonly IList<INeuralNetworkAxon> _outputAxons;

		public double Error => throw new NotImplementedException();

		public double Output => 1d;

		public NeuralNetworkBias()
		{
			_outputAxons = new List<INeuralNetworkAxon>();
		}

		public void AddInput(INeuralNetworkAxon axon) =>
			throw new InvalidOperationException("Bias can not have input axon");

		public void AddOutput(INeuralNetworkAxon axon)
		{
			if (axon == null)
				throw new ArgumentNullException(nameof(axon));

			_outputAxons.Add(axon);
		}

		public void AdjustWeights()
		{
			foreach (var axon in _outputAxons)
			{
				axon.AdjustWeight();
			}
		}

		public void BackpropagateError() =>
			throw new InvalidOperationException("Bias can not backpropagate error");

		public void BackpropagateError(double target) =>
			throw new InvalidOperationException("Bias can not backpropagate error");

		public void CalculateOutput() =>
			throw new InvalidOperationException("Bias can not calculate output");

		public void SetInput(double input) =>
			throw new InvalidOperationException("Bias can not accept input value");
	}
}