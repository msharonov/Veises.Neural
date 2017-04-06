using System.Collections.Generic;
using Veises.Neural;

namespace Veises.Recurrent
{
	public sealed class RecurrentNetwork: NeuralNetwork
	{
		public RecurrentNetwork(
			IReadOnlyCollection<NeuronLayer> layers,
			IErrorFunction errorFunction)
			: base(layers, errorFunction)
		{

		}
	}
}