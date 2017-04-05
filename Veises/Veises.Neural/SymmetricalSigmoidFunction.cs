using System;

namespace Veises.Neural
{
	public sealed class SymmetricalSigmoidFunction: IActivationFunction
	{
		public double Activate(double sum) => (1d - Math.Exp(-sum)) / (1d + Math.Exp(-sum));
	}
}
