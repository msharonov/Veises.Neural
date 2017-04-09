using System;

namespace Veises.Neural
{
	public sealed class SigmoidFunction: IActivationFunction
	{
		public double Activate(double sum) => 1d / (1d + Math.Exp(-sum));

		public double Deactivate(double output) => output * (1d - output);
	}
}