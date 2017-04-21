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
			if (inputsCount < 1)
				throw new ArgumentException("Inputs count can't be less than 1");

			if (activationFunction == null)
				throw new ArgumentNullException(nameof(activationFunction));

			if (errorFunction == null)
				throw new ArgumentNullException(nameof(errorFunction));

			var inputs = new List<IPerceptronInput>();

			var inputErrorFunction = new DeltaFunction();

			for (var i = 0; i < inputsCount; i++)
			{
				var inputWeight = GetNextWeight();

				var input = new PerceptronInput(inputWeight, inputErrorFunction);

				inputs.Add(input);
			}

			var biasInputWeight = 1d;

			var biasInput = new PerceptronBiasInput(biasInputWeight, inputErrorFunction);

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

		public void AdjustWeights(double error)
		{
			foreach (var input in Inputs)
			{
				input.AdjustWeight(error);
			}
		}

		public double GetError(double desiredOutput)
		{
			var output = CalculateOutput();

			var sumActivation = _activationFunction.GetDerivative(output);

			var error = (desiredOutput - output) * sumActivation;

			Debug.WriteLine($"Error value: {error}");

			return error;
		}

		public void Learn(double desiredOutput)
		{
			var globalError = _errorFunction.Calculate(CalculateOutput(), desiredOutput);

			var epochCount = 0;

			while (globalError > Settings.Default.LearningTestAcceptance &&
				epochCount < Settings.Default.MaxLearningEpochsCount)
			{
				Debug.WriteLine($"Global error: {globalError}");

				var error = GetError(desiredOutput);

				AdjustWeights(error);

				globalError = _errorFunction.Calculate(CalculateOutput(), desiredOutput);

				Debug.WriteLine($"Learn epoch {epochCount} is done");

				epochCount++;
			}
		}

		public void Load(IEnumerable<double> inputValues)
		{
			if (inputValues == null)
				throw new ArgumentNullException(nameof(inputValues));

			var enumerator = inputValues.GetEnumerator();

			foreach (var input in Inputs)
			{
				if (!(input is PerceptronInput))
					continue;

				if (!enumerator.MoveNext())
					throw new ArgumentException("Input values number does not match to a total input number");

				input.SetInput(enumerator.Current);
			}
		}
	}
}