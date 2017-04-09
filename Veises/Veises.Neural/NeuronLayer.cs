using System;
using System.Collections.Generic;
using System.Linq;

namespace Veises.Neural
{
	public interface INeuralNetworkLayer
	{
		void AdjustWeights();

		void BackpropagateError();

		void CalculateOutputs();

		IReadOnlyCollection<double> GetOutputs();

		IReadOnlyCollection<NeuralNetworkNeuron> GetNeurons();

		void SetExpectedOutputs(double[] expectedOutputs);

		void SetInputs(double[] inputs);
	}

	public sealed class NeuronLayer: INeuralNetworkLayer
	{
		public readonly IList<NeuralNetworkNeuron> Neurons;

		private readonly Bias _bias;

		public NeuronLayerType LayerType { get; set; }



		public NeuronLayer(NeuronLayerType layerType, IEnumerable<NeuralNetworkNeuron> neurons, Bias bias)
		{
			if (neurons == null)
				throw new ArgumentNullException(nameof(neurons));

			_bias = bias ?? throw new ArgumentNullException(nameof(bias));

			Neurons = neurons.ToList();
			LayerType = layerType;
		}

		public void AdjustWeights()
		{
			foreach (var neuron in Neurons)
			{
				neuron.AdjustWeights();
			}
		}

		public void BackpropagateError()
		{
			foreach (var neuron in Neurons)
			{
				neuron.CalculateError();
			}
		}

		public void CalculateOutputs()
		{
			foreach (var perceptron in Neurons)
			{
				perceptron.CalculateOutput();
			}
		}

		public IReadOnlyCollection<double> GetOutputs() =>
			Neurons
				.Select(_ => _.Output)
				.ToArray();

		public IReadOnlyCollection<NeuralNetworkNeuron> GetNeurons() =>
			Neurons
				.ToArray();

		public void SetExpectedOutputs(double[] expectedOutputs)
		{
			if (expectedOutputs == null)
				throw new ArgumentNullException(nameof(expectedOutputs));

			if (Neurons.Count != expectedOutputs.Length)
				throw new ArgumentException("Expected output items count mismatch", nameof(expectedOutputs));

			for (var i = 0; i < Neurons.Count; i++)
			{
				Neurons[i].CalculateError(expectedOutputs[i] - Neurons[i].Output);
			}
		}

		public void SetInputs(double[] inputs)
		{
			if (inputs == null)
				throw new ArgumentNullException(nameof(inputs));

			if (Neurons.Count != inputs.Length)
				throw new ArgumentException("Input neurons count mismatch", nameof(inputs));

			var i = 0;

			foreach (var neuron in Neurons)
			{
				neuron.SetInput(inputs[i++]);
			}
		}
	}
}