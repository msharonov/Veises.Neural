namespace Veises.Neural
{
	public interface INeuralNetworkTrainer
	{
		void Load(INeuralNetwork neuralNetwork);

		void Train(params NetworkLearnCase[] learningCases);
	}
}