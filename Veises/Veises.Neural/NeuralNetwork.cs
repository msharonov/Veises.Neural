using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;

namespace Veises.Neural
{
	public class NeuralNetwork: INeuralNetwork
	{
		public IReadOnlyCollection<INeuralNetworkLayer> NeuronLayers { get; private set; }

		private readonly IActivationFunction _activationFunction;

		public NeuralNetwork(
			IReadOnlyCollection<INeuralNetworkLayer> layers,
			IActivationFunction activationFunction)
		{
			NeuronLayers = layers ?? throw new ArgumentNullException(nameof(layers));
			_activationFunction = activationFunction ?? throw new ArgumentNullException(nameof(activationFunction));
		}

		public void SetInputs(params double[] inputValues) =>
			NeuronLayers.First().SetInputs(inputValues);

		public IEnumerable<double> GetOutputs()
		{
			foreach (var layer in NeuronLayers.Skip(1))
			{
				layer.CalculateOutputs();
			}

			return NeuronLayers.Last().GetOutputs();
		}

		public double GetGlobalError(params double[] desiredOutputs)
		{
			if (desiredOutputs == null)
				throw new ArgumentNullException(nameof(desiredOutputs));

			var sum = 0d;

			var outputs = GetOutputs().ToArray();

			for (var i = 0; i < desiredOutputs.Length; i++)
			{
				sum += Math.Pow(desiredOutputs[i] - outputs[i], 2d);
			}

			var globalError = 0.5 * sum;

			Debug.WriteLine($"Network global error: {globalError}");

			return globalError;
		}

		public void Learn(params double[] desiredOutputs)
		{
			if (desiredOutputs == null)
				throw new ArgumentNullException(nameof(desiredOutputs));

			var outputLayer = NeuronLayers.Last();

			outputLayer.InitializeErrors(desiredOutputs);

			foreach (var hiddenLayer in NeuronLayers.Reverse().Skip(1))
			{
				hiddenLayer.BackpropagateError();
			}

			foreach (var layer in NeuronLayers)
			{
				layer.AdjustWeights();
			}
		}
	}
}