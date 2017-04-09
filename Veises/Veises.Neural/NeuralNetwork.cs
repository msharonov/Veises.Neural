using System;
using System.Collections.Generic;
using System.Linq;

namespace Veises.Neural
{
	public class NeuralNetwork: INeuralNetwork
	{
		public IReadOnlyCollection<INeuralNetworkLayer> NeuronLayers { get; private set; }

		private readonly IErrorFunction _errorFunction;

		public double GlobalError { get; private set; }

		public NeuralNetwork(IReadOnlyCollection<INeuralNetworkLayer> layers, IErrorFunction errorFunction)
		{
			NeuronLayers = layers ?? throw new ArgumentNullException(nameof(layers));
			_errorFunction = errorFunction ?? throw new ArgumentNullException(nameof(errorFunction));
		}

		public IEnumerable<double> GetOutputs(params double[] inputs)
		{
			NeuronLayers.First().SetInputs(inputs);

			foreach (var layer in NeuronLayers.Skip(1))
			{
				layer.CalculateOutputs();
			}

			return NeuronLayers.Last().GetOutputs();
		}

		public double GetGlobalError(double[] inputs, double[] desiredOutputs)
		{
			var outputs = GetOutputs(inputs);

			var outputSum = outputs.Sum();

			var desiredOutputsSum = desiredOutputs.Sum();

			// TODO: corrent function 0.5d * Sum(pow(input[i] - output[i], 2))
			return GlobalError = _errorFunction.Calculate(outputSum, desiredOutputsSum);
		}
	}
}