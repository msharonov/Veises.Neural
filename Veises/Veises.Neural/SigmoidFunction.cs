using System;

namespace Veises.Neural
{
	public sealed class SigmoidFunction: IActivationFunction
	{
		public double Activate(double sum, double bias) => 1d / (1 + Math.Exp(-sum - bias));
	}
}