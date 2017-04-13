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

			foreach (var learningCase in learningCases)
			{
				var iterationNumber = 1;

				do
				{
					_neuralNetwork.SetInputs(learningCase.Input);

					var globalError = _neuralNetwork.GetGlobalError(learningCase.Expected);

					if (globalError < Settings.Default.LearningTestAcceptance)
						break;

					Debug.WriteLine($"Training iteration {iterationNumber}");

					iterationNumber++;
				}
				while (true);
			}
		}
	}
}
