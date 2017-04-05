using System;

namespace Veises.Neural
{
	public sealed class StepFunction: IActivationFunction
	{
		private readonly double _threaold;

		public StepFunction(double thresold = 0d)
		{
			_threaold = thresold;
		}

		public double Activate(double sum) => sum > _threaold ? 1d : 0d;
	}
}