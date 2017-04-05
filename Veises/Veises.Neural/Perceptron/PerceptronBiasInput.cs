using System;
using Veises.Neural.Properties;

namespace Veises.Neural.Perceptron
{
	public sealed class PerceptronBiasInput: IPerceptronInput
	{
		public const double Bias = 1d;

		public double Weight { get; private set; }

		private readonly IErrorFunction _errorFunction;

		public PerceptronBiasInput(double weight, IErrorFunction errorFunction)
		{
			_errorFunction = errorFunction ?? throw new ArgumentNullException(nameof(errorFunction));

			Weight = weight;
		}

		public double CalculateOutput() => Bias * Weight;

		public void AdjustWeight(double desiredOutput)
		{
			var localOutput = CalculateOutput();

			var localError = _errorFunction.Calculate(localOutput, desiredOutput);

			Weight += Settings.Default.LearningRate * localError * Bias;
		}
	}
}