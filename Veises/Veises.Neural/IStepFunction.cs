using System;

namespace Veises.Neural
{
	public sealed class StepFunction: IActivationFunction
	{
		public double Activate(double sum) => sum > 0 ? 1d : 0d;
	}
}