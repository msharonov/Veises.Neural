using System;

namespace Veises.Neural
{
	public sealed class BipolarSigmoidActivationFunction: IActivationFunction
	{
		public double Activate(double sum) =>
			(2d / (1d + Math.Exp(-sum))) - 1d;

		public double GetDerivative(double output) =>
			output * (1 - output);
	}
}