using System;
using System.Collections.Generic;
using System.Diagnostics;
using Veises.Neural.Properties;

namespace Veises.Neural.Perceptron
{
	public sealed class BinaryPerceptron
	{
		public readonly IReadOnlyCollection<IPerceptronInput> Inputs;

		private readonly IActivationFunction _activationFunction;

		private readonly IErrorFunction _errorFunction;

		private static Random _random = new Random();

		public BinaryPerceptron(
			IReadOnlyCollection<IPerceptronInput> inputs,
			IActivationFunction activationFunction,
			IErrorFunction errorFunction)
		{
			Inputs = inputs ?? throw new ArgumentNullException(nameof(inputs));
			_activationFunction = activationFunction ?? throw new ArgumentNullException(nameof(activationFunction));
			_errorFunction = errorFunction ?? throw new ArgumentNullException(nameof(errorFunction));
		}

		public static BinaryPerceptron Create(
			int inputsCount,
			IActivationFunction activationFunction,
			IErrorFunction errorFunction)
		{
			if (inputsCount < 1d)
				throw new ArgumentException("Inputs count can't be less than 1");

			if (activationFunction == null)
				throw new ArgumentNullException(nameof(activationFunction));

			if (errorFunction == null)
				throw new ArgumentNullException(nameof(errorFunction));

			var inputs = new List<IPerceptronInput>();

			for (var i = 0; i < inputsCount; i++)
			{
				var inputWeight = GetNextWeight();

				var input = new PerceptronInput(inputWeight);

				inputs.Add(input);
			}

			var biasInputWeight = 1d;

			var biasInput = new PerceptronBiasInput(biasInputWeight);

			inputs.Add(biasInput);

			return new BinaryPerceptron(inputs, activationFunction, errorFunction);
		}

		private static double GetNextWeight() => _random.NextDouble();

		public double CalculateOutput()
		{
			var sum = 0d;

			foreach (var input in Inputs)
			{
				sum += input.CalculateOutput();
			}

			return _activationFunction.Activate(sum);
		}

		public void AdjustWeights(double globalError)
		{
			foreach (var input in Inputs)
			{
				input.AdjustWeight(globalError);
			}
		}

		public double GetGlobalError(double desiredOutput)
		{
			var output = CalculateOutput();

			var globalError = desiredOutput - output;

			Debug.WriteLine($"Global error: {globalError}");

			return globalError;
		}

		public void Learn(double desiredOutput)
		{
			var globalError = GetGlobalError(desiredOutput);

			while (Math.Abs(globalError) > Settings.Default.LearningTestAcceptance)
			{
				AdjustWeights(globalError);

				globalError = GetGlobalError(desiredOutput);
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