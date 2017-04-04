using System;
using System.Collections.Generic;
using Veises.Neural.Properties;

namespace Veises.Neural.Perceptron
{
	public sealed class BinaryPerceptron
	{
		public readonly IReadOnlyCollection<IPerceptronInput> Inputs;

		public Func<double, double> ActivationFunc = _ => _ > Settings.Default.ActivationThreshold ? 1 : 0;

		public BinaryPerceptron(IReadOnlyCollection<IPerceptronInput> inputs)
		{
			Inputs = inputs ?? throw new ArgumentNullException(nameof(inputs));
		}

		public static BinaryPerceptron Create(int inputsCount, bool addBias = true)
		{
			if (inputsCount < 1)
				throw new ArgumentException("Inputs count can't be less than 1");

			var random = new Random();

			var inputs = new List<IPerceptronInput>();

			for (var i = 0; i < inputsCount; i++)
			{
				var inputWeight = random.Next(-1, 1);

				var input = new PerceptronInput(inputWeight);

				inputs.Add(input);
			}

			if (addBias)
			{
				var biasInputWeight = random.Next(-1, 1);

				var biasInput = new PerceptronBiasInput(biasInputWeight);

				inputs.Add(biasInput);
			}

			return new BinaryPerceptron(inputs);
		}

		public double CalculateOutput()
		{
			var sum = .0d;

			foreach (var input in Inputs)
			{
				sum += input.CalculateOutput();
			}

			return ActivationFunc.Invoke(sum);
		}

		private double GetLocalError(double desiredOutput)
		{
			var output = CalculateOutput();

			var localError = desiredOutput - output;

			return localError;
		}

		public void AdjustWeights(double desiredOutput)
		{
			var localError = GetLocalError(desiredOutput);

			foreach (var input in Inputs)
			{
				input.AdjustWeight(localError);
			}
		}

		public void Learn(double desiredOutput)
		{
			var localError = 1.0d;

			while (localError > Settings.Default.LearningTestAcceptance)
			{
				AdjustWeights(desiredOutput);

				localError = GetLocalError(desiredOutput);
			}
		}

		public void Load(IEnumerable<double> inputValues)
		{
			if (inputValues == null)
				throw new ArgumentNullException(nameof(inputValues));

			var enumerator = inputValues.GetEnumerator();

			enumerator.MoveNext();

			foreach (var input in Inputs)
			{
				var perceptronInput = input as PerceptronInput;

				if (perceptronInput == null)
					throw new ApplicationException("Incorrect data");

				perceptronInput.SetInput(enumerator.Current);

				if (!enumerator.MoveNext())
					break;
			}
		}
	}
}