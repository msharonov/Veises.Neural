using System;
using Veises.Neural;

namespace Veises.Recurrent
{
	public sealed class TanhActivationFunction: IActivationFunction
	{
		public double Activate(double sum) => Math.Tanh(sum);
	}
}