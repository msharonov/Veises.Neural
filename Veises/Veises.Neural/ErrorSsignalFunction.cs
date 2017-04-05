using System;

namespace Veises.Neural
{
	public sealed class ErrorSignalFunction: IErrorFunction
	{
		public double Calculate(double output, double target) =>
			(target - output) * output * (1 - output);
	}
}