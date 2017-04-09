using System;

namespace Veises.Neural
{
	public sealed class BipolarSigmoidActivationFunction: IActivationFunction
	{
		public double Activate(double sum) =>
			(2d / (1d + Math.Exp(-sum))) - 1d;

		public double Deactivate(double output) =>
			0.5d * (1 + output) * (1 - output);
	}
}