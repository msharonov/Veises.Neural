using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using Veises.Neural.Properties;

namespace Veises.Neural
{
	public sealed class NeuralNetwork
	{
		private readonly IReadOnlyCollection<NeuronLayer> _neuronLayers;

		private readonly IErrorFunction _errorFunction;

		private const int MinimalLayersCount = 3;

		public double GlobalError { get; private set; }

		public NeuralNetwork(IReadOnlyCollection<NeuronLayer> layers, IErrorFunction errorFunction)
		{
			_neuronLayers = layers ?? throw new ArgumentNullException(nameof(layers));
			_errorFunction = errorFunction ?? throw new ArgumentNullException(nameof(errorFunction));
		}

		public static NeuralNetwork Create(int[] layerNeuronCount, IErrorFunction errorFunction)
		{
			if (layerNeuronCount == null)
				throw new ArgumentNullException(nameof(layerNeuronCount));

			if (errorFunction == null)
				throw new ArgumentNullException(nameof(errorFunction));

			if (layerNeuronCount.Length < MinimalLayersCount)
				throw new ArgumentException(
					$"Neuron layers count can not be less than {MinimalLayersCount}, but found {layerNeuronCount.Length}.",
					nameof(layerNeuronCount));

			var layers = new List<NeuronLayer>();

			for (var layerNumber = 0; layerNumber < layerNeuronCount.Length; layerNumber++)
			{
				var layerType = NeuronLayerType.Hidden;

				if (layerNumber == 0)
					layerType = NeuronLayerType.Input;
				else if (layerNumber == layerNeuronCount.Length - 1)
					layerType = NeuronLayerType.Output;

				var layer = NeuronLayer.Create(layerType, layerNeuronCount[layerNumber]);

				layers.Add(layer);

				if (layerNumber > 0)
				{
					var previousLayer = layers[layerNumber - 1];

					Axon.Create(previousLayer, layer);
				}
			}

			return new NeuralNetwork(layers, errorFunction);
		}

		public IEnumerable<double> GetOutputs(params double[] inputs)
		{
			_neuronLayers.First().SetInputs(inputs);

			foreach (var layer in _neuronLayers.Skip(1))
			{
				layer.CalculateOutputs();
			}

			return _neuronLayers.Last().Outputs;
		}

		public double GetGlobalError(double[] inputs, double[] desiredOutputs)
		{
			var outputs = GetOutputs(inputs);

			var outputSum = outputs.Sum();

			var desiredOutputsSum = desiredOutputs.Sum();

			return GlobalError = _errorFunction.Calculate(outputSum, desiredOutputsSum);
		}

		public void Learn(params double[] expectedOutputs)
		{
			_neuronLayers.Last().SetExpectedOutputs(expectedOutputs);

			foreach (var layer in _neuronLayers.Reverse().Skip(1))
			{
				layer.BackpropagateError();
			}

			foreach (var layer in _neuronLayers)
			{
				layer.AdjustWeights();
			}
		}

		public void Learn(params NetworkLearnCase[] learnCases)
		{
			if (learnCases == null)
				throw new ArgumentNullException(nameof(learnCases));

			var iterationCount = 1;

			while (true)
			{
				var requireRepeatLearn = false;

				foreach (var learnCase in learnCases)
				{
					var outputs = GetOutputs(learnCase.Input).ToList();

					var isExpectedEqualsOutput = true;

					for (var i = 0; i < learnCase.Expected.Length; i++)
					{
						var diff = Math.Abs(learnCase.Expected[i] - outputs[i]);

						var isValueEaquals = diff < Settings.Default.LearningTestAcceptance;

						if (isValueEaquals == false)
							isExpectedEqualsOutput = false;
					}

					if (!isExpectedEqualsOutput)
					{
						Learn(learnCase.Expected);

						requireRepeatLearn = true;
					}
				}

				if (requireRepeatLearn == false)
					break;

				iterationCount++;
			}

			Debug.WriteLine($"Learn iterations total count: {iterationCount}");
		}
	}
}