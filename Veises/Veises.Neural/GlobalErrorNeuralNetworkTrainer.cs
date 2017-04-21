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
					_neuralNetwork.SetInputs(learnCase.Input);

					var globalError = _neuralNetwork.GetGlobalError(learnCase.Expected);

					var isExpectedEqualsOutput = globalError < Settings.Default.LearningTestAcceptance;

					if (!isExpectedEqualsOutput)
					{
						_neuralNetwork.Learn(learnCase.Expected);

						requireRepeat = true;
					}

					globalErrorSum += globalError;
				}

				Debug.WriteLine($"Learn iterations total count: {iterationCount}, global error: {globalErrorSum}");

				if (!requireRepeat)
					break;

				iterationCount++;

				if (iterationCount > 1000)
				{
					Debug.WriteLine("Maximal iterations count exeed. Learning process is stopped");

					break;
				}
			}
		}
	}
}
