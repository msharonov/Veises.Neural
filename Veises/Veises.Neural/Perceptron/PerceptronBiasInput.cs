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

		public void AdjustWeight(double error)
		{
			var localOutput = CalculateOutput();

			Weight += Settings.Default.LearningRate * Bias * error;
		}

		public void SetInput(double input) =>
			throw new InvalidOperationException("Bias can't take value");
	}
}