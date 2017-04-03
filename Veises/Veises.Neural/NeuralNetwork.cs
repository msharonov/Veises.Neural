using System;
using System.Collections.Generic;
using Veises.Neural.Properties;

namespace Veises.Neural
{
	public sealed class NeuralNetwork
	{
		private readonly IList<NeuronLayer> _neuronLayers;

		public NeuralNetwork(int[] layerNeuronsCount)
		{
			_neuronLayers = new List<NeuronLayer>();

			CreateLayers(layerNeuronsCount);
		}

		private void CreateLayers(int[] layerNeuronCount)
		{
			if (layerNeuronCount == null)
				throw new ArgumentNullException(nameof(layerNeuronCount));

			if (layerNeuronCount.Length < 3)
				throw new ArgumentException(
					$"Neuron layers count can not be less than 3, but found {layerNeuronCount.Length}.",
					nameof(layerNeuronCount));

			_neuronLayers.Clear();

			for (var layerNumber = 0; layerNumber < layerNeuronCount.Length; layerNumber++)
			{
				var layerType = NeuronLayerType.Hidden;

				if (layerNumber == 0)
					layerType = NeuronLayerType.Input;
				else if (layerNumber == layerNeuronCount.Length - 1)
					layerType = NeuronLayerType.Output;

				var layer = NeuronLayer.Create(layerType, layerNeuronCount[layerNumber]);

				_neuronLayers.Add(layer);

				if (layerNumber > 0)
				{
					var previousLayer = _neuronLayers[layerNumber - 1];

					Axon.Create(previousLayer, layer);
				}
			}
		}

		public double[] GetOutputs(params double[] inputs)
		{
			_neuronLayers[0].SetInputs(inputs);

			for (var i = 1; i < _neuronLayers.Count; i++)
			{
				_neuronLayers[i].CalculateOutputs();
			}

			return _neuronLayers[_neuronLayers.Count - 1].Outputs;
		}

		public void Learn(params double[] expectedOutputs)
		{
			_neuronLayers[_neuronLayers.Count - 1].SetExpectedOutputs(expectedOutputs);

			for (var i = _neuronLayers.Count - 2; i > 0; i--)
			{
				_neuronLayers[i].BackpropagateError();
			}

			for (var i = 0; i < (_neuronLayers.Count - 1); i++)
			{
				_neuronLayers[i].AdjustWeights();
			}
		}

		public void Learn(NetworkLearnCase[] learnCases)
		{
			if (learnCases == null)
				throw new ArgumentNullException(nameof(learnCases));

			while (true)
			{
				var requireRepeatLearn = false;

				foreach (var learnCase in learnCases)
				{
					var outputs = GetOutputs(learnCase.Input);

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
			}
		}
	}
}