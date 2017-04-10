using System;
using Veises.Neural.Properties;

namespace Veises.Neural
{
	public class NeuralNetworkAxon: INeuralNetworkAxon
	{
		private readonly INeuralNetworkNeuron _inputNeuron;

		private readonly INeuralNetworkNeuron _outputNeuron;

		public virtual double Weight { get; protected set; } = _random.NextDouble();

		public virtual double WeightedError => _outputNeuron.Error * Weight;

		private static readonly Random _random = new Random();

		private double _delta = 0d;

		public NeuralNetworkAxon(INeuralNetworkNeuron parent, INeuralNetworkNeuron child)
		{
			_inputNeuron = parent ?? throw new ArgumentNullException(nameof(parent));
			_outputNeuron = child ?? throw new ArgumentNullException(nameof(child));
		}

		public virtual void AdjustWeight()
		{
			_delta = Settings.Default.LearningRate
				* _outputNeuron.Error
				* _inputNeuron.Output +
				_delta * Settings.Default.Momentum;

			Weight += _delta;
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
	}
}