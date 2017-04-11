namespace Veises.Neural
{
	public interface INeuralNetworkNeuron
	{
		double Error { get; }

		double Output { get; }

		void AddInput(INeuralNetworkAxon axon);

		void AddOutput(INeuralNetworkAxon axon);

		void AdjustWeights();

		void CalculateError();

		void CalculateError(double target);

		void CalculateOutput();

		void SetInput(double input);
	}
}