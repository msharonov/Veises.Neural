using System;
using System.Diagnostics;
using Veises.Neural.Properties;

namespace Veises.Neural
{
	public class NeuralNetworkAxon: INeuralNetworkAxon
	{
		private readonly INeuralNetworkNeuron _inputNeuron;

		private readonly INeuralNetworkNeuron _outputNeuron;

		public virtual double Weight { get; protected set; } = _random.NextDouble();

		private static readonly Random _random = new Random();

		private readonly bool _isWithMomentum;

		private double _delta;

		public NeuralNetworkAxon(
			INeuralNetworkNeuron parent,
			INeuralNetworkNeuron child,
			bool isWithMomentum = false)
		{
			_inputNeuron = parent ?? throw new ArgumentNullException(nameof(parent));
			_outputNeuron = child ?? throw new ArgumentNullException(nameof(child));

			_isWithMomentum = isWithMomentum;
		}

		public virtual void AdjustWeight()
		{
			if (_isWithMomentum)
			{
				_delta = Settings.Default.LearningRate
					* _outputNeuron.Error
					* _inputNeuron.Output +
					_delta * Settings.Default.Momentum;
			}
			else
			{
				_delta = Settings.Default.LearningRate
					* _outputNeuron.Error
					* _inputNeuron.Output;
			}

			Weight -= _delta;

			if (Math.Abs(Weight) > 1d)
			{
				Debug.WriteLine($"Axon with weight {Weight} is overlearned");
			}
		}

		public static INeuralNetworkAxon Create(INeuralNetworkNeuron input, INeuralNetworkNeuron output)
		{
			if (input == null)
				throw new ArgumentNullException(nameof(input));
			if (output == null)
				throw new ArgumentNullException(nameof(output));

			var axon = new NeuralNetworkAxon(input, output);

			input.AddOutput(axon);

			output.AddInput(axon);

			return axon;
		}

		public static void Create(INeuralNetworkLayer inputLayer, INeuralNetworkLayer outputLayer)
		{
			if (inputLayer == null)
				throw new ArgumentNullException(nameof(inputLayer));
			if (outputLayer == null)
				throw new ArgumentNullException(nameof(outputLayer));

			foreach (var parentNexon in inputLayer.GetNeurons())
			{
				foreach (var childNexon in outputLayer.GetNeurons())
				{
					Create(parentNexon, childNexon);
				}
			}
		}

		public INeuralNetworkNeuron GetInputNeuron() => _inputNeuron;

		public double GetOutput() => _inputNeuron.Output * Weight;

		public INeuralNetworkNeuron GetOutputNeuron() => _outputNeuron;

		public virtual double GetWeightedError() => _outputNeuron.Error * Weight;
	}
}