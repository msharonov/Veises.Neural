using System;

namespace Veises.Neural
{
	public sealed class Bias
	{
		public readonly double Weight;

		private static Random _random = new Random();

		public Bias() : this(_random.NextDouble())
		{

		}

		public Bias(double value)
		{
			Weight = value;
		}
	}
}