using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;

namespace Veises.Neural
{
	public class NeuralNetworkLayer: INeuralNetworkLayer
	{
		protected readonly IReadOnlyCollection<INeuralNetworkNeuron> Neurons;

		private readonly INeuralNetworkNeuron _biasNeuron;

		public readonly NeuronLayerType LayerType;

		public NeuralNetworkLayer(
			NeuronLayerType layerType,
			IReadOnlyCollection<INeuralNetworkNeuron> neurons,
			INeuralNetworkNeuron biasNeuron)
		{
			if (neurons == null)
				throw new ArgumentNullException(nameof(neurons));

			if (layerType == NeuronLayerType.Input &&
				_biasNeuron != null)
				throw new InvalidOperationException("Input layer can not have bias");

			_biasNeuron = biasNeuron;

			Neurons = neurons.ToList();
			LayerType = layerType;

			Debug.WriteLine($"Neural network layer with type '{GetType().Name}' was created");
		}

		public virtual void AdjustWeights()
		{
			if (_biasNeuron != null)
				_biasNeuron.AdjustWeights();

			foreach (var neuron in Neurons)
			{
				neuron.AdjustWeights();
			}
		}

		public void InitializeErrors(IReadOnlyCollection<double> desiredOutput)
		{
			if (desiredOutput == null)
				throw new ArgumentNullException(nameof(desiredOutput));

			if (LayerType != NeuronLayerType.Output)
				throw new ApplicationException("Errors can't be initialized for non-output layers");

			var desiedOuputEnumerator = desiredOutput.GetEnumerator();

			foreach (var neuron in Neurons)
			{
				if (desiedOuputEnumerator.MoveNext())
					neuron.BackpropagateError(desiedOuputEnumerator.Current);
			}
		}

		public virtual void BackpropagateError()
		{
			foreach (var neuron in Neurons)
			{
				neuron.BackpropagateError();
			}
		}

		public virtual void CalculateOutputs()
		{
			foreach (var neuron in Neurons)
			{
				neuron.CalculateOutput();
			}
		}

		public virtual IReadOnlyCollection<double> GetOutputs() =>
			Neurons
				.Select(_ => _.Output)
				.ToList();

		public IReadOnlyCollection<INeuralNetworkNeuron> GetNeurons() => Neurons;

		public void SetInputs(IReadOnlyCollection<double> inputs)
		{
			if (inputs == null)
				throw new ArgumentNullException(nameof(inputs));

			if (Neurons.Count != inputs.Count)
				throw new ArgumentException("Input neurons count mismatch", nameof(inputs));

			var inputEnumerator = inputs.GetEnumerator();

			inputEnumerator.MoveNext();

			foreach (var neuron in Neurons)
			{
				if (inputEnumerator.MoveNext())
					neuron.SetInput(inputEnumerator.Current);
			}
		}
	}
}