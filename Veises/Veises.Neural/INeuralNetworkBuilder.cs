namespace Veises.Neural
{
	public interface INeuralNetworkBuilder
	{
		INeuralNetwork Build(int[] layerNeuronsCount, IActivationFunction activationFunction);
	}
}