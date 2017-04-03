using System;

namespace Veises.Recurrent
{
	public class RecurrentNetworkOld
	{
		private int numInput;
		private int numHidden;
		private int numOutput;

		private double[] inputs;

		private double[][] ihWeights; // input-hidden
		private double[] hBiases;
		private double[] hNodes; // hidden nodes

		private double[][] chWeights; // context-hidden
		private double[] cNodes; // context nodes

		private double[][] hoWeights; // hidden-output
		private double[] oBiases;

		private double[] outputs;
		private Random rnd;

		public RecurrentNetworkOld(int numInput, int numHidden,
		  int numOutput, int seed)
		{
			this.numInput = numInput;
			this.numHidden = numHidden;
			this.numOutput = numOutput;

			inputs = new double[numInput];

			ihWeights = MakeMatrix(numInput, numHidden);
			hBiases = new double[numHidden];
			hNodes = new double[numHidden];

			chWeights = MakeMatrix(numHidden, numHidden);
			cNodes = new double[numHidden];

			hoWeights = MakeMatrix(numHidden, numOutput);
			oBiases = new double[numOutput];
			outputs = new double[numOutput];

			rnd = new Random(seed);

			InitializeWeights();
			InitializeContext();
		}

		private static double[][] MakeMatrix(int rows, int cols)
		{
			var result = new double[rows][];

			for (var r = 0; r < result.Length; ++r)
				result[r] = new double[cols];

			return result;
		}

		private void InitializeWeights()
		{
			var numWeights = (numInput * numHidden) +
			  (numHidden * numOutput) +
			  (numHidden * numHidden) + numHidden + numOutput;

			var initialWeights = new double[numWeights];

			for (var i = 0; i < initialWeights.Length; ++i)
				initialWeights[i] = (0.001 - 0.0001) * rnd.NextDouble() + 0.0001;

			SetWeights(initialWeights);
		}

		private void InitializeContext()
		{
			for (var c = 0; c < numHidden; ++c)
				cNodes[c] = 0.3 + (0.1 * c);
		}

		public void SetWeights(double[] weights)
		{
			var numWeights = (numInput * numHidden) +
			  (numHidden * numOutput) +
			  (numHidden * numHidden) + numHidden + numOutput;

			if (weights.Length != numWeights)
				throw new Exception("Bad weights array in SetWeights");

			var p = 0;

			for (var i = 0; i < numInput; ++i)
				for (var j = 0; j < numHidden; ++j)
					ihWeights[i][j] = weights[p++];

			for (var i = 0; i < numHidden; ++i)
				hBiases[i] = weights[p++];

			for (var j = 0; j < numHidden; ++j)
				for (var k = 0; k < numOutput; ++k)
					hoWeights[j][k] = weights[p++];

			for (var k = 0; k < numOutput; ++k)
				oBiases[k] = weights[p++];

			for (var c = 0; c < numHidden; ++c)
				for (var j = 0; j < numHidden; ++j)
					chWeights[c][j] = weights[p++];
		}

		public double[] GetWeights()
		{
			var numWeights = (numInput * numHidden) +
			  (numHidden * numOutput) +
			  (numHidden * numHidden) + numHidden + numOutput;

			var result = new double[numWeights];

			var k = 0;

			for (var i = 0; i < ihWeights.Length; ++i)
				for (var j = 0; j < ihWeights[0].Length; ++j)
					result[k++] = ihWeights[i][j];

			for (var i = 0; i < hBiases.Length; ++i)
				result[k++] = hBiases[i];

			for (var i = 0; i < hoWeights.Length; ++i)
				for (var j = 0; j < hoWeights[0].Length; ++j)
					result[k++] = hoWeights[i][j];

			for (var i = 0; i < oBiases.Length; ++i)
				result[k++] = oBiases[i];
			return result;
		}

		public void SetContext(double[] values)
		{
			if (values.Length != numHidden)
				throw new Exception("Bad array in SetContext");

			for (var c = 0; c < numHidden; ++c)
				cNodes[c] = values[c];
		}

		public double[] ComputeOutputs(double[] xValues)
		{
			var hSums = new double[numHidden]; // hidden nodes sums scratch array
			var oSums = new double[numOutput]; // output nodes sums

			for (var i = 0; i < xValues.Length; ++i) // copy x-values to inputs
				inputs[i] = xValues[i];

			for (var j = 0; j < numHidden; ++j)  // compute i-h sum of weights * inputs
				for (var i = 0; i < numInput; ++i)
					hSums[j] += inputs[i] * ihWeights[i][j]; // note +=

			// add in context nodes * c-h weights
			for (var j = 0; j < numHidden; ++j)
				for (var c = 0; c < numHidden; ++c) // all c nodes contribute to each h node
					hSums[j] += cNodes[c] * chWeights[c][j]; // note +=

			for (var j = 0; j < numHidden; ++j)  // add biases to input-to-hidden sums
				hSums[j] += hBiases[j];

			for (var j = 0; j < numHidden; ++j)   // apply activation
				hNodes[j] = HyperTan(hSums[j]); // hard-coded

			for (var k = 0; k < numOutput; ++k)   // compute h-o sum of weights * hOutputs
				for (var j = 0; j < numHidden; ++j)
					oSums[k] += hNodes[j] * hoWeights[j][k];

			for (var k = 0; k < numOutput; ++k)  // add biases to input-to-hidden sums
				oSums[k] += oBiases[k];

			var softOut = Softmax(oSums); // all outputs at once for efficiency
			for (var k = 0; k < numOutput; ++k)
				outputs[k] = softOut[k];

			// copy h node value to corresponding c node
			for (var j = 0; j < numHidden; ++j)
				cNodes[j] = hNodes[j];

			var retResult = new double[numOutput]; // could define a GetOutputs 
			for (var k = 0; k < numOutput; ++k)
				retResult[k] = outputs[k];
			return retResult;
		} // ComputeOutputs

		private static double HyperTan(double x)
		{
			if (x < -20.0)
				return -1.0; // approximation is correct to 30 decimals
			else if (x > 20.0)
				return 1.0;
			else
				return Math.Tanh(x);
		}

		private static double[] Softmax(double[] oSums)
		{
			var result = new double[oSums.Length];

			var sum = 0.0;
			for (var k = 0; k < oSums.Length; ++k)
				sum += Math.Exp(oSums[k]);

			for (var k = 0; k < oSums.Length; ++k)
				result[k] = Math.Exp(oSums[k]) / sum;

			return result;
		}
	}
}
