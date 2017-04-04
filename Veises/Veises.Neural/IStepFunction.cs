using System;

namespace Veises.Neural
{
	public sealed class IStepFunction: IActivationFunction
	{
		public double Activate(double sum, double bias)
			=> sum > 0 ? 1d : 0d;
	}
}