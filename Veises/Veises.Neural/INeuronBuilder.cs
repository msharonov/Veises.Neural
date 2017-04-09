namespace Veises.Neural
{
	public interface INeuronBuilder
	{
		NeuralNetworkNeuron Build(Bias bias);
	}
}