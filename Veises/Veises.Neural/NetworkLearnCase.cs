using System;

namespace Veises.Neural
{
	public class NetworkLearnCase
	{
		public readonly double[] Input;

		public readonly double[] Expected;

		public NetworkLearnCase(double[] input, double[] expected)
		{
			Input = input ?? throw new ArgumentNullException(nameof(input));
			Expected = expected ?? throw new ArgumentNullException(nameof(expected));
		}
	}
}