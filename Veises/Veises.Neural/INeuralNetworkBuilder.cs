namespace Veises.Neural
{
	public interface INeuralNetworkBuilder
	{
		INeuralNetwork Build(
			IActivationFunction activationFunction,
			params int[] layerNeuronsCount);
	}
}