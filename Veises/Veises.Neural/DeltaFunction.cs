using System;

namespace Veises.Neural
{
	public sealed class DeltaFunction: IErrorFunction
	{
		public double Calculate(double output, double target) => target - output;
	}
}
