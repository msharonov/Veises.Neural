using System.Collections.Generic;

namespace Veises.Neural
{
	public interface INeuralNetworkLayer
	{
		void AdjustWeights();

		void BackpropagateError();

		void CalculateOutputs();

		IReadOnlyCollection<double> GetOutputs();

		IReadOnlyCollection<INeuralNetworkNeuron> GetNeurons();

		void InitializeErrors(params double[] desiredOutput);

		void SetInputs(double[] inputs);
	}
}