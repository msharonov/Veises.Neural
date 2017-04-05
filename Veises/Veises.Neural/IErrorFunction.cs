using System;

namespace Veises.Neural
{
	public interface IErrorFunction
	{
		double Calculate(double output, double target);
	}

	public sealed class ErrorFunction: IErrorFunction
	{
		public double Calculate(double output, double target) =>
			0.5d * Math.Pow(output - target, 2.0d);
	}
}