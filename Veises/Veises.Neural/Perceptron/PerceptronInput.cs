using Veises.Neural.Properties;

namespace Veises.Neural.Perceptron
{
	public sealed class PerceptronInput: IPerceptronInput
	{
		public double Input { get; private set; }

		public double Weight { get; private set; }

		public PerceptronInput(double weight, double input = .0d)
		{
			Weight = weight;
			Input = input;
		}

		public double CalculateOutput() => Input * Weight;

		public void SetInput(double input) => Input = input;

		public void AdjustWeight(double localError) =>
			Weight += Settings.Default.LearningRate * localError * Input;
	}
}