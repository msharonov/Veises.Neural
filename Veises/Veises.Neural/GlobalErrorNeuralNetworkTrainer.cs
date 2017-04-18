using System;
using System.Diagnostics;
using Veises.Neural.Properties;

namespace Veises.Neural
{
	public sealed class GlobalErrorNeuralNetworkTrainer: INeuralNetworkTrainer
	{
		private INeuralNetwork _neuralNetwork;

		public void Load(INeuralNetwork neuralNetwork) =>
			_neuralNetwork = neuralNetwork ?? throw new ArgumentNullException(nameof(neuralNetwork));

		public void Train(params NetworkLearnCase[] learningCases)
		{
			if (learningCases == null)
				throw new ArgumentNullException(nameof(learningCases));

			if (_neuralNetwork == null)
				throw new InvalidOperationException("Neural network is not initialized");

			var iterationCount = 1;

			while (true)
			{
				var requireRepeat = false;

				var globalErrorSum = 0d;

				foreach (var learnCase in learningCases)
				{
					var isExpectedEqualsOutput = true;

					_neuralNetwork.SetInputs(learnCase.Input);

					var globalError = _neuralNetwork.GetGlobalError(learnCase.Expected);

					var isValueEaquals = globalError < Settings.Default.LearningTestAcceptance;

					if (isValueEaquals == false)
						isExpectedEqualsOutput = false;

					if (!isExpectedEqualsOutput)
					{
						_neuralNetwork.Learn(learnCase.Expected);

						requireRepeat = true;
					}

					globalErrorSum += globalError;
				}

				if (!requireRepeat)
					break;

				Debug.WriteLine($"Learn iterations total count: {iterationCount}, global error: {globalErrorSum}");

				iterationCount++;

				if (iterationCount > 1000)
					break;
			}
		}
	}
}
