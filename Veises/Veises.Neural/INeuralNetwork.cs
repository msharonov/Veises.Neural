using System.Collections.Generic;

namespace Veises.Neural
{
	public interface INeuralNetwork
	{
		IReadOnlyCollection<NeuronLayer> NeuronLayers { get; }

		double GetGlobalError(double[] inputs, double[] desiredOutputs);

		IEnumerable<double> GetOutputs(params double[] inputs);
	}
}