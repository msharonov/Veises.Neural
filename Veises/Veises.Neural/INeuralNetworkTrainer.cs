namespace Veises.Neural
{
	public interface INeuralNetworkTrainer<T>
		where T : class, INeuralNetwork
	{
		void Load(T neuralNetwork);

		void Train(params NetworkLearnCase[] learningCases);
	}
}