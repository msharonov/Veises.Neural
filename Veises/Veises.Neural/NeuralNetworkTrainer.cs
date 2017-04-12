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

			do
			{
				var requireRepeat = false;

				foreach (var learnCase in learningCases)
				{
					var isExpectedEqualsOutput = true;

					_neuralNetwork.SetInputs(learnCase.Input);

					var outputs = _neuralNetwork.GetOutputs().ToArray();

					_neuralNetwork.GetGlobalError(learnCase.Expected);

					for (var i = 0; i < learnCase.Expected.Length; i++)
					{
						var diff = Math.Abs(learnCase.Expected[i] - outputs[i]);

						Debug.WriteLine($"Diff: {diff}");

						var isValueEaquals = diff < Settings.Default.LearningTestAcceptance;

						if (isValueEaquals == false)
							isExpectedEqualsOutput = false;
					}

					if (!isExpectedEqualsOutput)
					{
						_neuralNetwork.Learn(learnCase.Expected);

						requireRepeat = true;
					}
				}

				if (!requireRepeat)
					break;

				Debug.WriteLine($"Learn iterations total count: {iterationCount}");

				iterationCount++;

			} while (true);
		}
	}
}