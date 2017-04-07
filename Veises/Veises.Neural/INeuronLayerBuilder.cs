namespace Veises.Neural
{
	public interface INeuronLayerBuilder
	{
		NeuronLayer Build(NeuronLayerType layerType, int neuronsCount);
	}
}