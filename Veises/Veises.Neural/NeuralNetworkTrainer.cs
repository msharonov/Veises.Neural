using System;
using System.Diagnostics;
using System.Linq;
using Veises.Neural.Properties;

namespace Veises.Neural
{
	public sealed class NeuralNetworkTrainer: INeuralNetworkTrainer
	{
		private INeuralNetwork _neuralNetwork;

		public void Load(INeuralNetwork neuralNetwork) =>
			_neuralNetwork = neuralNetwork ?? throw new ArgumentNullException(nameof(neuralNetwork));

		public void Train(params NetworkLearnCase[] learningCases)
		{
			if (learningCases == null)
				throw new ArgumentNullException(nameof(learningCases));

			var iterationCount = 1;

			while (true)
			{
				var requireRepeatLearn = false;

				foreach (var learnCase in learningCases)
				{
					var outputs = _neuralNetwork.GetOutputs(learnCase.Input).ToList();

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
						Train(learnCase.Expected);

						requireRepeatLearn = true;
					}
				}

				if (requireRepeatLearn == false)
					break;

				iterationCount++;
			}

			Debug.WriteLine($"Learn iterations total count: {iterationCount}");
		}

		private void Train(params double[] expectedOutputs)
		{
			_neuralNetwork.NeuronLayers.Last().SetExpectedOutputs(expectedOutputs);

			foreach (var layer in _neuralNetwork.NeuronLayers.Reverse().Skip(1))
			{
				layer.BackpropagateError();
			}

			foreach (var layer in _neuralNetwork.NeuronLayers)
			{
				layer.AdjustWeights();
			}
		}
	}
}