using System;
using System.Collections.Generic;
using System.Linq;

namespace Veises.Neural
{
	public sealed class NeuralNetworkLayer: INeuralNetworkLayer
	{
		public readonly IList<NeuralNetworkNeuron> Neurons;

		private readonly Bias _bias;

		public readonly NeuronLayerType LayerType;

		public NeuralNetworkLayer(NeuronLayerType layerType, IEnumerable<NeuralNetworkNeuron> neurons, Bias bias)
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

		public void InitializeErrors(params double [] desiredOutput)
		{
			if (desiredOutput == null)
				throw new ArgumentNullException(nameof(desiredOutput));

			if (LayerType != NeuronLayerType.Output)
				throw new ApplicationException("Errors can't be initialized for non-output layers");

			for (var i = 0; i <Neurons.Count; i++)
			{
				Neurons[i].CalculateError(desiredOutput[i]);
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