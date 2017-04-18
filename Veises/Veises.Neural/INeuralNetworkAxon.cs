namespace Veises.Neural
{
	public interface INeuralNetworkAxon
	{
		double Weight { get; }

		void AdjustWeight();

		double GetOutput();

		double GetWeightedError();

		INeuralNetworkNeuron GetInputNeuron();

		INeuralNetworkNeuron GetOutputNeuron();
	}
}