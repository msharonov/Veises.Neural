using System;

namespace Veises.Neural
{
	public sealed class Bias
	{
		public readonly double Value;

		private static Random _random = new Random();

		public Bias() : this(_random.Next(0, 1))
		{

		}

		public Bias(double value)
		{
			Value = value;
		}
	}
}