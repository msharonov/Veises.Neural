using System;
using Veises.Neural.Properties;

namespace Veises.Neural.Perceptron
{
	public sealed class PerceptronInput: IPerceptronInput
	{
		public double Input { get; private set; }

		public double Weight { get; private set; }

		private readonly IErrorFunction _errorFunction;

		public PerceptronInput(double weight, IErrorFunction errorFunction)
		{
			_errorFunction = errorFunction ?? throw new ArgumentNullException(nameof(errorFunction));

			Weight = weight;
		}

		public double CalculateOutput() => GetInputValue() * Weight;

		public void SetInput(double input)
		{
			if (input < 0d)
				throw new ArgumentException("Perception input value can't be less than 0.");

			Input = input;
		}

		private double GetInputValue() =>
			Input > Settings.Default.InputThreshold ? 1d : 0d;

		public void AdjustWeight(double error) =>
			Weight += Settings.Default.LearningRate * error * Input;
	}
}