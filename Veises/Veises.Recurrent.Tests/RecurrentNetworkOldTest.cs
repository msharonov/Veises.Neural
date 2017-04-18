using NUnit.Framework;

namespace Veises.Recurrent.Tests
{
	[TestFixture]
	public sealed class RecurrentNetworkOldTest
	{
		private RecurrentNetworkOld _recurrentNetwork;

		[Test]
		public void ShouldComputeOutputs()
		{
			var numInput = 2;
			var numHidden = 3;
			var numOutput = 2;

			var seed = 0;

			_recurrentNetwork = new RecurrentNetworkOld(numInput, numHidden, numOutput, seed);

			var wts = new double[] {
				0.01, 0.02, 0.03, 0.04, 0.05, 0.06,
				0.07, 0.08, 0.09, 0.10, 0.11, 0.12,
				0.13, 0.14, 0.15, 0.16, 0.17, 0.18,
				0.19, 0.20, 0.21, 0.22, 0.23, 0.24,
				0.25, 0.26 };

			_recurrentNetwork.SetWeights(wts);

			var cValues = new double[] { 0.3, 0.4, 0.5 };

			_recurrentNetwork.SetContext(cValues);

			var xValues = new double[] { 1.0, 2.0 };

			var yValues = _recurrentNetwork.ComputeOutputs(xValues);
		}
	}
}