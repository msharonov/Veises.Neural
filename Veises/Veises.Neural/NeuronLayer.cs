using System;
using System.Collections.Generic;
using System.Linq;

namespace Veises.Neural
{
	public sealed class NeuronLayer
	{
		public readonly IList<Neuron> Neurons;

		public NeuronLayerType LayerType { get; set; }

		public IEnumerable<double> Outputs => Neurons.Select(_ => _.Output).ToArray();

		public NeuronLayer(NeuronLayerType layerType, IEnumerable<Neuron> neurons)
		{
			if (neurons == null)
				throw new ArgumentNullException(nameof(neurons));

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

		public static NeuronLayer Create(NeuronLayerType layerType, int neuronsCount)
		{
			if (neuronsCount < 1)
				throw new ArgumentException("Layer neurons count can not be less than 1.");

			var neurons = Enumerable
				.Range(0, neuronsCount)
				.Select(_ => new Neuron(new SigmoidFunction()));

			return new NeuronLayer(layerType, neurons);
		}

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