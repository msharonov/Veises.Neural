namespace Veises.Neural
{
	public interface INeuralNetworkBuilder<T>
		where T : class, INeuralNetwork
	{
		T Build(int[] layerNeuronsCount, IErrorFunction errorFunction);
	}
}