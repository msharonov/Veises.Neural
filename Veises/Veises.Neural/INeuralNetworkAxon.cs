namespace Veises.Neural
{
	public interface INeuralNetworkAxon
	{
		double Weight { get; }

		double WeightedError { get; }

		void AdjustWeight();

		double GetOutput();

		INeuralNetworkNeuron GetInputNeuron();

		INeuralNetworkNeuron GetOutputNeuron();
	}
}