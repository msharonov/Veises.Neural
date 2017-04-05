using FluentAssertions;
using NUnit.Framework;
using Veises.Neural.Perceptron;

namespace Veises.Neural.Tests.Perceptron
{
	[TestFixture]
	public sealed class BinaryPerceptronTest
	{
		private static double[] Three = new[]
			{
				1.0d, 1.0d, 1.0d,
				0.0d, 0.0d, 1.0d,
				1.0d, 1.0d, 1.0d,
				0.0d, 0.0d, 1.0d,
				1.0d, 1.0d, 1.0d
			};

		private static double[] Four = new[]
	{
				1.0d, 0.0d, 1.0d,
				1.0d, 0.0d, 1.0d,
				1.0d, 1.0d, 1.0d,
				0.0d, 0.0d, 1.0d,
				0.0d, 0.0d, 1.0d
			};

		[Test]
		public void ShouldRecognizeDigit()
		{
			var perceptron = BinaryPerceptron.Create(
				15,
				new StepFunction(0d),
				new ErrorFunction());

			perceptron.Load(Three);

			perceptron.Learn(1d);

			perceptron.Load(Four);

			perceptron.Learn(0d);

			var result = perceptron.CalculateOutput();

			result.ShouldBeEquivalentTo(0d);

			perceptron.Load(Three);

			result = perceptron.CalculateOutput();

			result.ShouldBeEquivalentTo(1d);
		}
	}
}