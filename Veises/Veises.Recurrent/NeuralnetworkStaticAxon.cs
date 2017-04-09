using Veises.Neural;

namespace Veises.Recurrent
{
	public sealed class NeuralnetworkStaticAxon: NeuralNetworkAxon
	{
		public override double Weight => 1d;

		public override double WeightedError => 0d;

		public NeuralnetworkStaticAxon(
			INeuralNetworkNeuron parent,
			INeuralNetworkNeuron child)
			: base(parent, child)
		{

		}

		public override void AdjustWeight()
		{
			// do nothing
		}
	}
}