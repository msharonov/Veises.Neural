using System.Collections.Generic;

namespace Veises.Neural
{
	public interface INeuralNetwork
	{
		IReadOnlyCollection<INeuralNetworkLayer> NeuronLayers { get; }

		void Learn(params double[] desiredOutputs);

		double GetGlobalError(params double[] desiredOutputs);

		IReadOnlyCollection<double> GetOutputs();

		void SetInputs(params double[] inputValues);
	}
}