using Veises.Neural;

namespace Veises.Recurrent
{
	public sealed class NeuralnetworkStaticAxon: NeuralNetworkAxon
	{
		public override double Weight => 1d;

		public NeuralnetworkStaticAxon(
			INeuralNetworkNeuron parent,
			INeuralNetworkNeuron child,
			double weight = 1d)
			: base(parent, child)
		{
			Weight = weight;
		}

		public override void AdjustWeight()
		{
			// do nothing
		}
	}
}