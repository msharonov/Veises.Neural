using System;
using Veises.Neural.Properties;

namespace Veises.Neural
{
	public sealed class Axon
	{
		private readonly Neuron _inputNeuron;

		private readonly Neuron _outputNeuron;

		public double Weight { get; private set; }

		public double WeightedError => _outputNeuron.Error * Weight;

		private double _delta { get; set; }

		private static readonly Random Rand = new Random();

		public double WeightedOutput => _inputNeuron.Output * Weight;

		public Axon(Neuron parent, Neuron child)
		{
			_inputNeuron = parent ?? throw new ArgumentNullException(nameof(parent));
			_outputNeuron = child ?? throw new ArgumentNullException(nameof(child));

			Weight = Rand.NextDouble() - 0.5;
		}

		public void AdjustWeight()
		{
			_delta = Settings.Default.LearningRate
				* _outputNeuron.Error
				* _inputNeuron.Output +
				_delta * Settings.Default.Momentum;

			Weight += _delta;
		}

		public static Axon Create(Neuron input, Neuron output)
		{
			if (input == null)
				throw new ArgumentNullException(nameof(input));
			if (output == null)
				throw new ArgumentNullException(nameof(output));

			var axon = new Axon(input, output);

			input.AddOutput(axon);

			output.AddInput(axon);

			return axon;
		}

		public static void Create(NeuronLayer inputLayer, NeuronLayer outputLayer)
		{
			if (inputLayer == null)
				throw new ArgumentNullException(nameof(inputLayer));
			if (outputLayer == null)
				throw new ArgumentNullException(nameof(outputLayer));

			foreach (var parentNexon in inputLayer.Neurons)
			{
				foreach (var childNexon in outputLayer.Neurons)
				{
					Axon.Create(parentNexon, childNexon);
				}
			}
		}
	}
}