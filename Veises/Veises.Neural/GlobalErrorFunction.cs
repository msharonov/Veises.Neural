﻿using System;

namespace Veises.Neural
{
	public sealed class GlobalErrorFunction: IErrorFunction
	{
		public double Calculate(double output, double target) =>
			0.5d * Math.Pow(target - output, 2.0d);
	}
}