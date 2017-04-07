using System;

namespace Veises.Neural
{
	public sealed class NeuronBuilder: INeuronBuilder
	{
		private readonly IActivationFunction _activationFunction;

		public NeuronBuilder(IActivationFunction activationFunction)
		{
			_activationFunction = activationFunction ?? throw new ArgumentNullException(nameof(activationFunction));
		}

		public Neuron Build(Bias layerBias)
		{
			if (layerBias == null)
				throw new ArgumentNullException(nameof(layerBias));

			return new Neuron(_activationFunction, layerBias);
		}
	}
}