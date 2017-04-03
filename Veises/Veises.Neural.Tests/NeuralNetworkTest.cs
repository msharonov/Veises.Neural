using System;
using NUnit.Framework;
using FluentAssertions;

namespace Veises.Neural.Tests
{
	[TestFixture]
	public class NeuralNetworkTest
	{
		private NeuralNetwork _target;

		private static readonly double[] OneInput =
		{
			0, 1, 0,
			0, 1, 0,
			0, 1, 0,
			0, 1, 0,
			1, 1, 1
		};

		private static readonly double[] OneOutput =
		{
			1, 0, 0
		};

		private static readonly double[] FourInput =
		{
			1, 0, 1,
			1, 0, 1,
			1, 1, 1,
			0, 0, 1,
			0, 0, 1
		};

		private static readonly double[] FourOutput =
		{
			0, 1, 0
		};

		private static readonly double[] FiveInput =
		{
			1, 1, 1,
			1, 0, 0,
			1, 1, 1,
			0, 0, 1,
			1, 1, 1,
		};

		private static readonly double[] FiveOutput =
		{
			0, 0, 1
		};

		public static object[] TestCases =
		{
			new object[] {OneInput, OneOutput},
			new object[] {FourInput, FourOutput},
			new object[] {FiveInput, FiveOutput}
		};

		[SetUp]
		public void SetUp()
		{
			_target = NeuralNetwork.Create(new[] { 15, 50, 50, 3 });

			var learnCases = new[]
			{
				new NetworkLearnCase(OneInput, OneOutput),
				new NetworkLearnCase(FourInput, FourOutput),
				new NetworkLearnCase(FiveInput, FiveOutput)
			};

			_target.Learn(learnCases);
		}

		[TestCaseSource(nameof(TestCases))]
		public void ShouldRecognizeSymbolsViaNet(double[] input, double[] output)
		{
			var result = _target.GetOutputs(input);

			for (var i = 0; i < output.Length; i++)
			{
				Math.Abs(output[i] - result[i])
					.Should()
					.BeLessOrEqualTo(0.1d);
			}
		}
	}
}