namespace Veises.Neural
{
	public interface INeuralNetworkLayerBuilder
	{
		INeuralNetworkLayer Build(NeuronLayerType layerType, int neuronsCount);
	}
}